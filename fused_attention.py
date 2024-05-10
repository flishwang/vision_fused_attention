"""
Vision Fused Attention
* Copyright 2022- flish_wang @ SGRI, SGCC Cop. Ltd. 
* Email: flish_wang@sina.com

* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files
* (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""


"""
Fused Attention

/*
* Copyright 2018-2020 Philippe Tillet
* Copyright 2020-2022 OpenAI
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files
* (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""
import warnings

import torch,torch.compiler

import triton
import triton.language as tl

# We don't run auto-tuning every time to keep the tutorial fast. Uncommenting
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.

attn_fwd_cache ={
    (16, 64, 4, -4, False, 8, 0): (2, 4),
    (16, 64, 4, -4, True, 8, 0): (2, 4),
    (16, 64, -4, 4, False, 8, 0): (2, 4),
    (16, 64, -4, 4, True, 8, 0): (2, 4),
    (64, 64, 1, 1, False, 8, 0): (4, 4),
    (64, 64, 1, 1, True, 8, 0): (8, 2),
    (64, 64, 3, 3, False, 8, 0): (4, 4),
    (64, 64, 3, 3, True, 8, 0): (4, 4),
    (64, 64, 5, 5, False, 8, 0): (4, 2),
    (64, 64, 5, 5, True, 8, 0): (4, 6),
    (256, 64, 1, 1, False, 8, 0): (4, 4),
    (256, 64, 1, 1, True, 8, 0): (2, 6),
    (16, 64, 4, -4, False, 7, 5): (2, 4),
    (16, 64, 4, -4, True, 7, 5): (2, 4),
    (16, 64, -4, 4, False, 7, 5): (2, 4),
    (16, 64, -4, 4, True, 7, 5): (2, 4),
    (64, 64, 1, 1, False, 7, 5): (4, 9),
    (64, 64, 1, 1, True, 7, 5): (4, 6),
    (64, 64, 3, 3, False, 7, 5): (4, 6),
    (64, 64, 3, 3, True, 7, 5): (4, 9),
    (64, 64, 5, 5, False, 7, 5): (4, 4),
    (64, 64, 5, 5, True, 7, 5): (4, 2),
    (256, 64, 1, 1, False, 7, 5): (4, 6),
    (256, 64, 1, 1, True, 7, 5): (4, 2),
}

attn_bwd_pre_cache= {
    (16, 64, 8, 0): (2, 4),
    (64, 64, 8, 0): (8,4),
    (256, 64, 8, 0): (4,9),
    (16, 64, 7, 5): (2, 4),
    (64, 64, 7, 5): (2,2),
    (256, 64, 7, 5): (2,4),
}
attn_bwd_cache={
    (16, 64, 4, -4, False, 8, 0): (2, 2),
    (16, 64, 4, -4, True, 8, 0): (2, 2),
    (16, 64, -4, 4, False, 8, 0): (2, 2),
    (16, 64, -4, 4, True, 8, 0): (2, 2),
    (64, 64, 1, 1, True, 8, 0): (8, 6),
    (64, 64, 3, 3, True, 8, 0): (4, 4),
    (64, 64, 5, 5, True, 8, 0): (4, 9),
    (256, 64, 1, 1, True, 8, 0): (4, 9),
    (64, 64, 1, 1, False, 8, 0): (2, 4),
    (64, 64, 3, 3, False, 8, 0): (4, 4),
    (64, 64, 5, 5, False, 8, 0): (4, 2),
    (256, 64, 1, 1, False, 8, 0): (4, 9),
    (16, 64, 4, -4, False, 7, 5): (2, 2),
    (16, 64, 4, -4, True, 7, 5): (2, 2),
    (16, 64, -4, 4, False, 7, 5): (2, 2),
    (16, 64, -4, 4, True, 7, 5): (2, 2),
    (64, 64, 1, 1, False, 7, 5): (4, 2),
    (64, 64, 1, 1, True, 7, 5): (4, 2),
    (64, 64, 3, 3, False, 7, 5): (4, 6),
    (64, 64, 3, 3, True, 7, 5): (4, 2),
    (64, 64, 5, 5, False, 7, 5): (4, 9),
    (64, 64, 5, 5, True, 7, 5): (4, 6),
    (256, 64, 1, 1, False, 7, 5): (4, 2),
    (256, 64, 1, 1, True, 7, 5): (4, 2),
}
_capability = torch.cuda.get_device_capability()

# #
# @triton.autotune(
#     configs=[
#         triton.Config(dict(), num_stages=6, num_warps=2),
#         triton.Config(dict(), num_stages=6, num_warps=4),
#         triton.Config(dict(), num_stages=4, num_warps=2),
#         triton.Config(dict(), num_stages=4, num_warps=4),
#         triton.Config(dict(), num_stages=2, num_warps=2),
#         triton.Config(dict(), num_stages=2, num_warps=4),
#     ],
#     key=['qs4','qs5','window_size1','window_size2','with_bias'],
# )
@triton.jit
def _attn_fwd(
        Q, K, V, B, sm_scale, M, Out,  #
        qd0,qd1,qd2,qd3,qd4,qd5,  # B,nH,h,w,S,C
        kd0,kd1,kd2,kd3,kd4,kd5,
        vd0,vd1,vd2,vd3,vd4,vd5,
        od0,od1,od2,od3,od4,od5,
        bd0,bd1,bd2,bd3,bd4,bd5, # B,nH,h,w,S,QS
        qs0,qs1: tl.constexpr,
        qs2,qs3,
        qs4: tl.constexpr, # N_CTX, S
        qs5: tl.constexpr, # BLOCK_D
        ks1: tl.constexpr, #
        bs0,bs1,bs2,bs3,bs4,bs5,
        window_size1: tl.constexpr,
        window_size2: tl.constexpr,
        BLOCK_M: tl.constexpr,
        with_bias: tl.constexpr,
):
    tl.static_assert(qs4 % BLOCK_M == 0)

    BLOCK_D: tl.constexpr = qs5
    start_s = tl.program_id(0)
    off_hw = tl.program_id(1)
    off_bH = tl.program_id(2)
    off_b = (off_bH // qs1).to(tl.int64)
    off_H = (off_bH % qs1).to(tl.int64)
    off_ht = (off_hw // qs3).to(tl.int64)
    off_wt = (off_hw % qs3).to(tl.int64)

    off_kH = off_H % ks1

    q_offset = off_b * qd0 + off_H * qd1
    k_offset = off_b * kd0 + off_kH * kd1
    v_offset = off_b * vd0 + off_kH * vd1
    o_offset = off_b * od0 + off_H * od1

    b_offset = (off_b % bs0) * bd0 + (off_H % bs1) * bd1

    mdwt = qs4
    mdht = mdwt * qs3
    mdH = mdht * qs2
    mdb = mdH * qs1
    off_m = start_s * BLOCK_M + off_wt * mdwt + off_ht * mdht + off_H * mdH + off_b * mdb
    M_ptrs = M + off_m + tl.arange(0,BLOCK_M)

    # initialize pointer to m and l
    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32) + 1.0
    acc = tl.zeros((BLOCK_M,BLOCK_D), dtype=tl.float32)
    # load scales
    qk_scale = sm_scale * 1.4426950408889634

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset + off_ht * qd2 + off_wt * qd3,  # locate to current B,H (block start location)
        shape=(qs4, BLOCK_D),  # shape inside the block, N,D
        strides=(qd4, qd5),  # stride of shape inside for N,D
        offsets=(start_s * BLOCK_M, 0),  # inner axis offset inside
        block_shape=(BLOCK_M, BLOCK_D),  # inner length of each axis
        order=(1, 0),
    )
    q = tl.load(Q_block_ptr)

    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset + off_ht * od2 + off_wt * od3,
        shape=(qs4, BLOCK_D),
        strides=(od4, od5),
        offsets=(start_s * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )  # same with Q


    if window_size1 < 0:
        loop_range1: tl.constexpr = -window_size1
        h_offset = off_ht % loop_range1
    elif window_size1 % 2 == 0:
        loop_range1: tl.constexpr = window_size1
        h_offset = (window_size1 -1) // 2 + (off_ht % 2 == 1)
    else:
        loop_range1: tl.constexpr = window_size1
        h_offset = (window_size1 -1) // 2

    if window_size2 < 0:
        loop_range2: tl.constexpr = -window_size2
        w_offset = off_wt % loop_range2
    elif window_size2 % 2 == 0:
        loop_range2: tl.constexpr = window_size2
        w_offset = (window_size2 -1) // 2 + (off_wt % 2 == 1)
    else:
        loop_range2: tl.constexpr = window_size2
        w_offset = (window_size2 -1) // 2

    for hidx in tl.static_range(0,loop_range1):
        for widx in tl.static_range(0,loop_range2):
            cur_h = off_ht + hidx - h_offset
            cur_w = off_wt + widx - w_offset
            for start_n in tl.static_range(0, qs4 // BLOCK_M):
                if cur_h >= 0 and cur_h < qs2:
                    if cur_w >= 0 and cur_w < qs3:
                        K_block_ptr = tl.make_block_ptr(
                            base=K + (k_offset + cur_h * kd2 + cur_w * kd3),
                            shape=(BLOCK_D, qs4),  # for D,N2
                            strides=(kd5, kd4),  # for D,N2
                            offsets=(0, start_n * BLOCK_M),
                            block_shape=(BLOCK_D, BLOCK_M),  # for D,N2
                            order=(0, 1),
                        )
                        V_block_ptr = tl.make_block_ptr(
                            base=V + (v_offset + cur_h * vd2 + cur_w * vd3),
                            shape=(qs4, BLOCK_D),  # for D,N2
                            strides=(vd4, vd5),  # for D,N2
                            offsets=(start_n * BLOCK_M, 0),
                            block_shape=(BLOCK_M, BLOCK_D),  # for D,N2
                            order=(1, 0),
                        )
                        k = tl.load(K_block_ptr)
                        qk = tl.dot(q, k)

                        if with_bias: # cur for kv and off for q
                            bq_off_n = hidx * loop_range2 + widx
                            if bs5 > bs4:
                                shape_bs5 = bs5
                            else:
                                shape_bs5 = bs4
                            B_block_ptr = tl.make_block_ptr(
                                base=B + (b_offset + (off_ht % bs2) * bd2 + (off_wt % bs3) * bd3),
                                shape=(qs4, shape_bs5),  # for D,N2
                                strides=(bd4, bd5),  # for D,N2
                                offsets=(start_s * BLOCK_M, (start_n * BLOCK_M + bq_off_n * qs4) % shape_bs5),
                                block_shape=(BLOCK_M, BLOCK_M),  # for D,N2
                                order=(1, 0),
                            )

                            bias = tl.load(B_block_ptr)
                            qk += bias

                        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
                        qk = qk * qk_scale - m_ij[:, None]
                        p = tl.math.exp2(qk)
                        l_ij = tl.sum(p, 1)
                        # -- update m_i and l_i
                        alpha = tl.math.exp2(m_i - m_ij)
                        l_i = l_i * alpha + l_ij
                        # -- update output accumulator --
                        acc = acc * alpha[:, None]
                        # update acc
                        v = tl.load(V_block_ptr)
                        p = p.to(v.dtype)
                        acc = tl.dot(p, v, acc)
                        m_i = m_ij

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    tl.store(M_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))

# # #
# @triton.autotune(
#     configs=[
#         triton.Config(dict(), num_stages=9, num_warps=2),
#         triton.Config(dict(), num_stages=9, num_warps=4),
#         triton.Config(dict(), num_stages=9, num_warps=8),
#         triton.Config(dict(), num_stages=4, num_warps=2),
#         triton.Config(dict(), num_stages=4, num_warps=4),
#         triton.Config(dict(), num_stages=4, num_warps=8),
#     ],
#     key=['qs4','qs5'],
# )
@triton.jit
def _attn_bwd_preprocess(O, DO,  #
                         Delta,  #
                         od0, od1, od2, od3, od4: tl.constexpr, od5: tl.constexpr,
                         qs0, qs1: tl.constexpr,
                         qs2, qs3, qs4: tl.constexpr, qs5: tl.constexpr,
                         BLOCK_M: tl.constexpr,
                         ):
    BLOCK_D: tl.constexpr = qs5
    start_s = tl.program_id(0)
    off_hw = tl.program_id(1)
    off_bH = tl.program_id(2)
    off_b = (off_bH // qs1).to(tl.int64)
    off_H = (off_bH % qs1).to(tl.int64)
    off_ht = (off_hw // qs3).to(tl.int64)
    off_wt = (off_hw % qs3).to(tl.int64)

    o_offset = off_b * od0 + off_H * od1

    mdht = qs4 * qs3
    mdH = mdht * qs2
    mdb = mdH * qs1
    offs_m = start_s * BLOCK_M + off_wt * qs4 + off_ht * mdht + off_H * mdH + off_b * mdb
    Delta_ptrs = Delta + offs_m + tl.arange(0, BLOCK_M)

    o_block_ptr = tl.make_block_ptr(
        base=O + o_offset + off_ht * od2 + off_wt * od3,
        shape=(qs4, BLOCK_D),
        strides=(od4, od5),
        offsets=(start_s * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    do_block_ptr = tl.make_block_ptr(
        base=DO + o_offset + off_ht * od2 + off_wt * od3,
        shape=(qs4, BLOCK_D),
        strides=(od4, od5),
        offsets=(start_s * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )

    o = tl.load(o_block_ptr)
    do = tl.load(do_block_ptr).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta_ptrs, delta)

#
# @triton.autotune(
#     configs=[
#         triton.Config(dict(), num_stages=6, num_warps=8),
#         triton.Config(dict(), num_stages=4, num_warps=2),
#         triton.Config(dict(), num_stages=4, num_warps=4),
#         triton.Config(dict(), num_stages=4, num_warps=8),
#         triton.Config(dict(), num_stages=2, num_warps=2),
#         triton.Config(dict(), num_stages=2, num_warps=4),
#         triton.Config(dict(), num_stages=2, num_warps=8),
#     ],
#     key=['qs4','qs5','window_size1','window_size2','with_bias'],
# )
@triton.jit
def _attn_bwd(Q, K, V, Bias, sm_scale,  #
              DO, DQ, DK, DV,  #
              M, Delta,
              qd0, qd1, qd2, qd3, qd4, qd5,  # B,nH,h,w,S,C
              kd0, kd1, kd2, kd3, kd4, kd5,
              vd0, vd1, vd2, vd3, vd4, vd5,
              od0, od1, od2, od3, od4, od5,
              bd0, bd1, bd2, bd3, bd4, bd5,  # B,nH,h,w,S,QS
              qs0, qs1: tl.constexpr,
              qs2, qs3,
              qs4: tl.constexpr,  # N_CTX, S
              qs5: tl.constexpr,  # BLOCK_D
              ks1: tl.constexpr,  #
              bs0, bs1, bs2, bs3, bs4, bs5,
              window_size1: tl.constexpr,
              window_size2: tl.constexpr,
              BLOCK_M: tl.constexpr,
              with_bias: tl.constexpr,
              qk_scale: tl.constexpr,
              ):

    BLOCK_D: tl.constexpr = qs5
    start_s = tl.program_id(0)
    off_hw = tl.program_id(1)
    off_bH = tl.program_id(2)
    off_b = (off_bH // qs1).to(tl.int64)
    off_H = (off_bH % qs1).to(tl.int64)
    off_ht = (off_hw // qs3).to(tl.int64)
    off_wt = (off_hw % qs3).to(tl.int64)

    off_kH = off_H % ks1

    q_offset = off_b * qd0 + off_H * qd1
    k_offset = off_b * kd0 + off_kH * kd1
    v_offset = off_b * vd0 + off_kH * vd1
    o_offset = off_b * od0 + off_H * od1
    b_offset = (off_b % bs0) * bd0 + (off_H % bs1) * bd1

    mdwt = qs4
    mdht = mdwt * qs3
    mdH = mdht * qs2
    mdb = mdH * qs1

    offs_m = off_H * mdH + off_b * mdb + tl.arange(0, BLOCK_M)

    ## ======== compute dk and dv =========
    dv = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    dk = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    K_block_ptr = tl.make_block_ptr(
        base=K + (k_offset + off_ht * kd2 + off_wt * kd3),
        shape=(qs4, qs5),  # for D,N2
        strides=(kd4, kd5),  # for D,N2
        offsets=(start_s * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),  # for D,N2
        order=(1,0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + (v_offset + off_ht * vd2 + off_wt * vd3),
        shape=(qs4, qs5),  # for D,N2
        strides=(vd4, vd5),  # for D,N2
        offsets=(start_s * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),  # for D,N2
        order=(1, 0),
    )


    k = (tl.load(K_block_ptr) * qk_scale).to(K.type.element_ty)
    v = tl.load(V_block_ptr)

    if window_size1 < 0:
        loop_range1: tl.constexpr = -window_size1
        h_offset = off_ht % loop_range1
    elif window_size1 % 2 == 0:
        loop_range1: tl.constexpr = window_size1
        h_offset = (window_size1 - 1) // 2 + (off_ht % 2 == 0)
    else:
        loop_range1: tl.constexpr = window_size1
        h_offset = (window_size1 - 1) // 2

    if window_size2 < 0:
        loop_range2: tl.constexpr = -window_size2
        w_offset = off_wt % loop_range2
    elif window_size2 % 2 == 0:
        loop_range2: tl.constexpr = window_size2
        w_offset = (window_size2 - 1) // 2 + (off_wt % 2 == 0)
    else:
        loop_range2: tl.constexpr = window_size2
        w_offset = (window_size2 - 1) // 2

    for hidx in tl.static_range(0, loop_range1): # cur for q and off for kv
        for widx in tl.static_range(0, loop_range2):
            cur_h = off_ht + hidx - h_offset
            cur_w = off_wt + widx - w_offset
            for start_n in tl.static_range(0, qs4 // BLOCK_M):
                if cur_h >= 0 and cur_h < qs2:
                    if cur_w >= 0 and cur_w < qs3:
                        Qt_block_ptr = tl.make_block_ptr(
                            base=Q + q_offset + cur_h * qd2 + cur_w * qd3,
                            # locate to current B,H (block start location)
                            shape=(BLOCK_D, qs4),  # shape inside the block, N,D
                            strides=(qd5, qd4),  # stride of shape inside for N,D
                            offsets=(0, start_n * BLOCK_M),  # inner axis offset inside
                            block_shape=(BLOCK_D, BLOCK_M),  # inner length of each axis
                            order=(0, 1),
                        )
                        qT = tl.load(Qt_block_ptr)
                        m = tl.load(M + offs_m + (cur_h * mdht + cur_w * mdwt + start_n * BLOCK_M))
                        kqT = tl.dot(k, qT)  # QK

                        if with_bias:  # cur for q and off for kv
                            if window_size1 < 0:
                                bq_off_h = off_ht % loop_range1
                            elif window_size1 % 2 == 0:
                                bq_off_h = off_ht + (window_size1 // 2) - 1 - cur_h + (cur_h %2)
                            else:
                                bq_off_h = off_ht - cur_h + h_offset

                            if window_size2 < 0:
                                bq_off_w = off_wt % loop_range2
                            elif window_size2 % 2 == 0:
                                bq_off_w = off_wt + (window_size2 // 2) - 1 - cur_w + (cur_w % 2)
                            else:
                                bq_off_w = off_wt - cur_w + w_offset
                            bq_off_n = bq_off_h * loop_range2 + bq_off_w
                            if bs5 > bs4:
                                shape_bs5 = bs5
                            else:
                                shape_bs5 = bs4
                            Bt_block_ptr = tl.make_block_ptr(
                                base=Bias + (b_offset + (cur_h % bs2) * bd2 + (cur_w % bs3) * bd3),
                                shape=(shape_bs5, qs4),  # for D,N2
                                strides=(bd5, bd4),  # for D,N2
                                offsets=((start_s * BLOCK_M + bq_off_n * qs4) % shape_bs5, start_n * BLOCK_M),
                                block_shape=(BLOCK_M, BLOCK_M),  # for D,N2
                                order=(0, 1),
                            )
                            bias_t = tl.load(Bt_block_ptr)
                            kqT += bias_t * qk_scale

                        DO_block_ptr = tl.make_block_ptr(
                            base=DO + o_offset + cur_h * od2 + cur_w * od3,
                            # locate to current B,H (block start location)
                            shape=(qs4, BLOCK_D),  # shape inside the block, N,D
                            strides=(od4, od5),  # stride of shape inside for N,D
                            offsets=(start_n * BLOCK_M, 0),  # inner axis offset inside
                            block_shape=(BLOCK_M, BLOCK_D),  # inner length of each axis
                            order=(1, 0),
                        )
                        do = tl.load(DO_block_ptr)
                        pT = tl.math.exp2(kqT - m[None, :].to(kqT.dtype))
                        ppT = pT
                        ppT = ppT.to(DO.type.element_ty)
                        dv = tl.dot(ppT, do, dv)
                        # D (= delta) is pre-divided by ds_scale.
                        Di = tl.load(Delta + offs_m + (cur_h * mdht + cur_w * mdwt + start_n * BLOCK_M))
                        # Compute dP and dS.
                        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
                        dsT = pT * (dpT - Di[None, :])
                        dsT = dsT.to(Q.type.element_ty)
                        dk = tl.dot(dsT, tl.trans(qT), dk)

    DK_block_ptr = tl.make_block_ptr(
        base=DK + (o_offset + off_ht * od2 + off_wt * od3),
        shape=(qs4, qs5),  # for D,N2
        strides=(od4, od5),  # for D,N2
        offsets=(start_s * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),  # for D,N2
        order=(1, 0),
    )
    DV_block_ptr = tl.make_block_ptr(
        base=DV + (o_offset + off_ht * od2 + off_wt * od3),
        shape=(qs4, qs5),  # for D,N2
        strides=(od4, od5),  # for D,N2
        offsets=(start_s * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),  # for D,N2
        order=(1, 0),
    )

    dk *= sm_scale
    tl.store(DK_block_ptr, dk.to(DK.type.element_ty))
    tl.store(DV_block_ptr, dv.to(DV.type.element_ty))

    ## ======== compute dq =========

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset + off_ht * qd2 + off_wt * qd3,  # locate to current B,H (block start location)
        shape=(qs4, BLOCK_D),  # shape inside the block, N,D
        strides=(qd4, qd5),  # stride of shape inside for N,D
        offsets=(start_s * BLOCK_M, 0),  # inner axis offset inside
        block_shape=(BLOCK_M, BLOCK_D),  # inner length of each axis
        order=(1, 0),
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO + o_offset + off_ht * od2 + off_wt * od3,  # locate to current B,H (block start location)
        shape=(qs4, BLOCK_D),  # shape inside the block, N,D
        strides=(od4, od5),  # stride of shape inside for N,D
        offsets=(start_s * BLOCK_M, 0),  # inner axis offset inside
        block_shape=(BLOCK_M, BLOCK_D),  # inner length of each axis
        order=(1, 0),
    )
    q = (tl.load(Q_block_ptr) * qk_scale).to(Q.type.element_ty)
    do = tl.load(DO_block_ptr)
    dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    m = tl.load(M + offs_m + (off_ht * mdht + off_wt * mdwt + start_s * BLOCK_M))
    m = m[:, None]
    Di = tl.load(Delta + offs_m + (off_ht * mdht + off_wt * mdwt + start_s * BLOCK_M))


    if window_size1 < 0:
        h_offset = off_ht % loop_range1
    elif window_size1 % 2 == 0:
        h_offset = (window_size1 - 1) // 2 + (off_ht % 2 == 1)
    else:
        h_offset = (window_size1 - 1) // 2

    if window_size2 < 0:
        w_offset = off_wt % loop_range2
    elif window_size2 % 2 == 0:
        w_offset = (window_size2 - 1) // 2 + (off_wt % 2 == 1)
    else:
        w_offset = (window_size2 - 1) // 2

    for hidx in tl.static_range(0, loop_range1): # off for q and cur for kv
        for widx in tl.static_range(0, loop_range2):
            cur_h = off_ht + hidx - h_offset
            cur_w = off_wt + widx - w_offset
            for start_n in tl.static_range(0, qs4 // BLOCK_M):
                if cur_h >=0 and cur_h < qs2:
                    if cur_w >=0 and cur_w < qs3:

                        KT_block_ptr = tl.make_block_ptr(
                            base=K + (k_offset + cur_h * kd2 + cur_w * kd3),
                            shape=(BLOCK_D, qs4),  # for D,N2
                            strides=(kd5, kd4),  # for D,N2
                            offsets=(0, start_n * BLOCK_M),
                            block_shape=(BLOCK_D, BLOCK_M),  # for D,N2
                            order=(0, 1),
                        )
                        VT_block_ptr = tl.make_block_ptr(
                            base=V + (v_offset + cur_h * vd2 + cur_w * vd3),
                            shape=(BLOCK_D, qs4),  # for D,N2
                            strides=(vd5,vd4),  # for D,N2
                            offsets=(0,start_n * BLOCK_M),
                            block_shape=(BLOCK_D,BLOCK_M),  # for D,N2
                            order=(0,1),
                        )
                        kT = tl.load(KT_block_ptr)
                        vT = tl.load(VT_block_ptr)
                        qk = tl.dot(q, kT)

                        if with_bias:
                            bq_off_n = hidx * loop_range2 + widx
                            if bs5 > bs4:
                                shape_bs5 = bs5
                            else:
                                shape_bs5 = bs4
                            B_block_ptr = tl.make_block_ptr(
                                base=Bias + (b_offset + (off_ht % bs2) * bd2 + (off_wt % bs3) * bd3),
                                shape=(qs4, shape_bs5),  # for D,N2
                                strides=(bd4, bd5),  # for D,N2
                                offsets=(start_s * BLOCK_M, (start_n * BLOCK_M + bq_off_n * qs4) % shape_bs5),
                                block_shape=(BLOCK_M, BLOCK_M),  # for D,N2
                                order=(1, 0),
                            )
                            bias = tl.load(B_block_ptr)
                            qk += bias * qk_scale

                        p = tl.math.exp2(qk - m)
                        dp = tl.dot(do, vT).to(tl.float32)
                        ds = p * (dp - Di[:, None])
                        ds = ds.to(K.type.element_ty)
                        # Compute dQ.
                        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
                        dq = tl.dot(ds, tl.trans(kT), dq)

    DQ_block_ptr = tl.make_block_ptr(
        base=DQ + o_offset + off_ht * od2 + off_wt * od3,  # locate to current B,H (block start location)
        shape=(qs4, BLOCK_D),  # shape inside the block, N,D
        strides=(od4, od5),  # stride of shape inside for N,D
        offsets=(start_s * BLOCK_M, 0),  # inner axis offset inside
        block_shape=(BLOCK_M, BLOCK_D),  # inner length of each axis
        order=(1, 0),
    )
    # Write back dQ.
    dq *= sm_scale
    tl.store(DQ_block_ptr, dq.to(DQ.type.element_ty))

@torch.compiler.disable
def call_triton_func(func,grid,*args,**kwargs):
    return func.__getitem__(grid)(*args,**kwargs)

class _block_attention(torch.autograd.Function):

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
    @staticmethod
    def forward(ctx, q, k, v, b=None, window_size1=3, window_size2=3, sm_scale = 1.0, dropout = 0):
        # q: B,nH,h,w,S,C: batch_size, num_heads, height blocks, width blocks, block size, head dim
        # b: B,nH,h,w,S,QS: batch_size, num_heads, height blocks, width blocks, Q nearby blocks, S block size, S

        block_size = q.shape[4]
        S = q.shape[4]
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        assert Lq == Lk
        assert Lk in {16, 32, 64, 128, 256}
        assert q.shape[1] % k.shape[1] == 0 and q.dim() == 6, \
            'q must be 6-D tensors with bs,nH,ht,wt,ss,d, and nH must divided by k heads'

        if dropout > 0:
            mask = torch.rand(1, q.shape[1], q.shape[2], q.shape[3], 1,
                              S if b is None else b.shape[-1], device=q.device) < dropout
            mask = mask.expand(-1, -1, -1, -1, S, -1) * -9999.0
            if b is not None:
                b = b + mask
            else:
                b = mask

        if b is None:
            bs = (0,0,0,0,0,0)
            bd = (0,0,0,0,0,0)
            with_bias = False
        else:
            assert (b.shape[4] == block_size
                    and (b.shape[5] == window_size1 * window_size2 * block_size
                         or b.shape[5] == -window_size1 * window_size2 * block_size
                         or b.shape[5] == block_size)), 'bias shape mismatch'
            with_bias = True
            bs=b.size()
            bd=b.stride()

        o = torch.empty_like(q)
        capability = torch.cuda.get_device_capability()

        if capability[0] == 7:
            SM = 63000
        elif capability == (8,0):
            SM = 163000
        elif capability == (8,6):
            SM = 99000
        elif capability == (9,0):
            SM = 227000
        else:
            raise NotImplementedError
        BLOCK_M = min(64, S)

        num_warps,num_stages = attn_fwd_cache.get((S,Lk,window_size1,window_size2,with_bias,capability[0],capability[1]),
                                                  (4,4))

        grid = (triton.cdiv(q.shape[4], BLOCK_M), q.shape[2] * q.shape[3], q.shape[0] * q.shape[1])
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2], q.shape[3], q.shape[4]),
                        device=q.device, dtype=torch.float32)

        call_triton_func(_attn_fwd,grid,q, k, v, b, sm_scale, M, o,  #
            *q.stride(),  #
            *k.stride(),  #
            *v.stride(),
            *o.stride(),
            *bd,  #
            q.size(0), q.size(1), q.size(2), q.size(3),q.size(4),q.size(5),
            k.size(1),
            *bs,
            window_size1 = window_size1,
            window_size2 = window_size2,
            BLOCK_M=BLOCK_M,  #
            with_bias = with_bias,
            num_warps = num_warps,num_stages = num_stages)


        ctx.save_for_backward(q, k, v, b, o, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.Lk = Lk
        ctx.SM = SM
        ctx.bs = bs
        ctx.bd = bd
        ctx.capability = capability
        ctx.window_size1 = window_size1
        ctx.window_size2 = window_size2
        ctx.with_bias = with_bias
        return o

    @torch.cuda.amp.custom_bwd
    @staticmethod
    def backward(ctx, do):
        q, k, v, b, o, M = ctx.saved_tensors # o,M is the value output and maximum value of qk
        # q: B,nH,h,w,S,C: batch_size, num_heads, height blocks, width blocks, block size, head dim
        # k: B,nH2,h,w,S,C
        # b: B,nH,h,w,S,QS: batch_size, num_heads, height blocks, width blocks, Q nearby blocks, S block size, S
        assert o.stride() == do.stride(),f'stride mismatch, o {o.stride()} do {do.stride()}'
        dq = torch.empty_like(q)
        dk = torch.empty_like(q)
        dv = torch.empty_like(q)
        S = q.shape[4]
        PRE_BLOCK = min(128,S)
        BLOCK_M = 64
        BLOCK_M = min(BLOCK_M, S)

        num_warps, num_stages = attn_bwd_pre_cache.get((S, ctx.Lk, ctx.capability[0],ctx.capability[1]),(8, 4))

        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        grid = (S // PRE_BLOCK, q.shape[2] * q.shape[3], q.shape[0] * q.shape[1])
        delta = torch.empty_like(M)
        call_triton_func(_attn_bwd_preprocess,grid,o, do,  #
            delta,  #
            *o.stride(),
            *o.size(), # b,nH,h,w,S
            BLOCK_M=PRE_BLOCK,
            num_warps = num_warps, num_stages=num_stages
        )

        num_warps, num_stages = attn_bwd_cache.get((S, ctx.Lk, ctx.window_size1, ctx.window_size2, ctx.with_bias,
                                                    ctx.capability[0],ctx.capability[1]),(4, 2))

        grid = (S // BLOCK_M, q.shape[2] * q.shape[3], q.shape[0] * q.shape[1])
        call_triton_func(_attn_bwd,grid,q, k, v, b, ctx.sm_scale, do, dq, dk, dv,  #
            M, delta,  #
            *q.stride(), #
            *k.stride(),  #
            *v.stride(),
            *o.stride(),
            *ctx.bd,  #
            *q.size(),
            k.size(1),
            *ctx.bs,
            window_size1=ctx.window_size1,
            window_size2=ctx.window_size2,
            BLOCK_M=BLOCK_M,  #
            with_bias=ctx.with_bias,
            qk_scale = ctx.sm_scale * RCP_LN2,
            num_warps=num_warps,
            num_stages=num_stages
        )
        if q.size(1) > k.size(1):
            dk = dk.unflatten(1,(-1,k.size(1))).sum(1)
            dv = dv.unflatten(1, (-1, k.size(1))).sum(1)


        return dq, dk, dv, None, None, None, None, None


block_attention = _block_attention.apply

def test_op(Z, H, N_CTX, D_HEAD,ws=3,NB=2, ws2=None, attn_bias=True, dtype=torch.float16,sm_scale = 0.25, kv_group = 1,dropout=0.):
    if ws2 is None:
        ws2 = ws
    rws = ws
    rws2 = ws2
    ws = rws if rws >=0 else -rws
    ws2 = rws2 if rws2 >=0 else -rws2
    torch.manual_seed(20)
    qkv0 = torch.empty(Z,NB,NB,N_CTX,(H+H * 2 //kv_group),D_HEAD,
                       dtype=dtype,device='cuda').normal_(mean=0.0, std=0.5).requires_grad_()
    qkv = qkv0.permute(0,4,1,2,3,5)
    q = qkv[:,:H]
    k = qkv[:,H:H+H//kv_group]
    v = qkv[:,H+H//kv_group:]

    if attn_bias:
        bias = torch.empty((Z, H,NB,NB, N_CTX, ws*ws*N_CTX), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5)
    else:
        bias = None

    dout = torch.randn_like(q)
    # reference implementation

    kwargs = dict(Z=Z,H=H,N=N_CTX,D=D_HEAD,rws=rws,rws2=rws2,NB=NB,attn_bias=attn_bias,dtype=dtype,dropout=dropout,
                  sm=sm_scale,kv_group=kv_group)
    print(f'compare {kwargs}')

    torch.manual_seed(20)
    tri_out = block_attention(q, k, v, bias, rws, rws2, sm_scale,dropout)
    tri_out.backward(dout)
    tri_grad, qkv0.grad = qkv0.grad.clone(), None

    if NB * NB * N_CTX * NB * NB * N_CTX * Z * H >= 10000000:
        print('    large input, skip naive torch allclose test')
        return

    torch.manual_seed(20)
    bias1 = bias
    qkv = qkv0.permute(0, 4, 1, 2, 3, 5)
    q = qkv[:, :H]
    k = qkv[:, H:H + H // kv_group]
    v = qkv[:, H + H // kv_group:]
    q1 = q.reshape(Z,H,NB*NB*N_CTX,D_HEAD)
    k1 = k.reshape(Z,H//kv_group,NB*NB*N_CTX,D_HEAD).repeat(1,kv_group,1,1)
    v1 = v.reshape(Z,H//kv_group,NB*NB*N_CTX,D_HEAD).repeat(1,kv_group,1,1)
    if dropout > 0:
        mask = torch.rand(1, q.shape[1], q.shape[2], q.shape[3], 1,
                          N_CTX if bias is None else bias.shape[-1], device=q.device) < dropout
        mask = mask.expand(-1, -1, -1, -1, N_CTX, -1) * -9999.0
        if bias is not None:
            bias1 = bias1 + mask
        else:
            bias1 = mask


    b1 = torch.zeros((Z,H,NB*NB*N_CTX,NB*NB*N_CTX),dtype=dtype,device='cuda') - 9999.0
    with torch.no_grad():
        for qi in range(NB):
            for qj in range(NB):
                for hidx in range(ws):
                    for widx in range(ws2):
                        if rws < 0:
                            ki = qi // ws * ws + hidx
                        else:
                            ki = qi + hidx - ((ws-1) // 2) - ((ws % 2 == 0)  & (qi % 2 == 1))

                        if rws2 < 0:
                            kj = qj // ws2 * ws2 + widx
                        else:
                            kj = qj + widx - ((ws2 - 1) // 2) - ((ws2 % 2 == 0) & (qj % 2 == 1))

                        if ki >=0 and ki < NB and kj >=0 and kj < NB:
                            ql = qi * NB + qj
                            kl = ki * NB + kj
                            dl = hidx * ws2 + widx
                            if bias1 is not None:
                                b1[:, :, ql * N_CTX: (ql + 1) * N_CTX, kl * N_CTX: (kl + 1) * N_CTX] = \
                                    bias1[:, :, qi, qj, :, dl * N_CTX: (dl + 1) * N_CTX]
                            else:
                                b1[:, :, ql * N_CTX: (ql + 1) * N_CTX, kl * N_CTX: (kl + 1) * N_CTX] = 0

    p = (torch.matmul(q1,k1.transpose(-1,-2)) + b1) * sm_scale
    p = torch.softmax(p.float(), dim=-1).to(q1.dtype)
    ref_out = torch.matmul(p,v1).view(*q.shape)
    ref_out.backward(dout)
    ref_grad, qkv0.grad = qkv0.grad.clone(), None


    if not torch.allclose(ref_out, tri_out, atol=1e-3, rtol=0):
        print(f'    out mismatch, max diff {(ref_out-tri_out).abs().max().item()}')
        if not torch.allclose(ref_out, tri_out, atol=2e-2, rtol=0):
            raise ValueError()
        #pdb.set_trace()

    if not torch.allclose(ref_grad[...,:H,:], tri_grad[...,:H,:], atol=1e-3, rtol=0):
        print(f'    dv mismatch, max diff {(ref_grad[...,:H,:]- tri_grad[...,:H,:]).abs().max().item()}')
        #raise ValueError()
        #pdb.set_trace()

    if not torch.allclose(ref_grad[...,H:H+H//kv_group,:], tri_grad[...,H:H+H//kv_group,:], atol=1e-3, rtol=0):
        print(f'    dk mismatch, max diff '
              f'{(ref_grad[...,H:H+H//kv_group,:]-tri_grad[...,H:H+H//kv_group,:]).abs().max().item()}')
        #raise ValueError()
        #pdb.set_trace()

    if not torch.allclose(ref_grad[...,H+H//kv_group:,:], tri_grad[...,H+H//kv_group:,:], atol=1e-3, rtol=0):
        print(f'    dq mismatch, max diff '
              f'{(ref_grad[...,H+H//kv_group:,:]-tri_grad[...,H+H//kv_group:,:]).abs().max().item()}')


if __name__ == '__main__':
    test_op(2, 4, 16, 64, ws=-4, NB=8, attn_bias = False, dtype=torch.float16)
    test_op(2, 4, 16, 64, ws=-4, ws2 = 4, NB=8, dtype=torch.float16)
    test_op(2, 4, 16, 64, ws=4, ws2= -4, NB=8, dtype=torch.float16)
    test_op(2, 4, 16, 64, ws=4, NB=8, attn_bias=False, dtype=torch.float16)
    test_op(2, 4, 16, 64, ws=4, NB=8, dtype=torch.float16, dropout=0.25)

    test_op(4, 16, 64, 64, ws=1, NB=1, attn_bias=False, dtype=torch.float16)
    test_op(4, 16, 64, 64, ws=1, NB=1, dtype=torch.float16, dropout=0.25)
    test_op(4, 16, 64, 64, ws=3, NB=10, attn_bias=False, dtype=torch.float16)
    test_op(4, 16, 64, 64, ws=5, NB=10, attn_bias=False, dtype=torch.float16)
    test_op(4, 16, 16, 64, ws=4, ws2 = -4, attn_bias=False, NB=20, dtype=torch.float16)
    test_op(4, 16, 16, 64, ws=-4, ws2 = 4, attn_bias=False, NB=20, dtype=torch.float16)

    # test_op(2, 4, 64, 64, ws=3, NB=4, dtype=torch.float16, dropout=0.25)
    # test_op(2, 4, 16, 64, ws=5, NB=8, dtype=torch.float16)
    #
    # test_op(64, 16, 256, 64, ws=1, NB=1, dtype=torch.float16)
    # test_op(4, 16, 64, 64, ws=1, NB=2, attn_bias=False, dtype=torch.float16)
    # test_op(4, 16, 64, 64, ws=3, NB=10, attn_bias=False, dtype=torch.float16)
    # test_op(4, 16, 64, 64, ws=5, NB=10, attn_bias=False, dtype=torch.float16)
    # test_op(64, 16, 256, 64, ws=1, NB=1, attn_bias=False, dtype=torch.float16)
    # test_op(4, 16, 64, 64, ws=3, NB=10, attn_bias=False, dtype=torch.float16, kv_group=4)

    #test_op(2, 2, 128, 64, ws=1, NB=1, attn_bias=False, dtype=torch.float16)
    #test_op(2, 2, 128, 64, ws=1, NB=1, dtype=torch.float16, sm_scale=0.5)
    #test_op(2, 8, 128, 64, ws=1, NB=1, dtype=torch.float16, sm_scale=0.5, kv_group=2)
    test_op(2, 2, 128, 64, ws=1, NB=1, attn_bias=False, dtype=torch.float32)
    test_op(2, 2, 128, 64, ws=1, NB=1, dtype=torch.float32, sm_scale= 1.0)

    #test_op(4, 8, 64, 64, ws=1, NB=2, dtype=torch.float16)
    #test_op(4, 8, 64, 64, ws=1, NB=1, dtype=torch.float32, sm_scale=0.125)
    #test_op(4, 8, 64, 64, ws=1, NB=1, attn_bias=False, dtype=torch.float16)
    #test_op(4, 8, 64, 64, ws=1, NB=1, attn_bias=False, dtype=torch.float32)

    test_op(2, 4, 64, 64, ws=3, NB=2, dtype=torch.float16)


    #test_op(2, 4, 128, 64, ws=3, NB=2, dtype=torch.float16)
    #test_op(2, 16, 128, 64, ws=3, NB=2, dtype=torch.float16, kv_group= 4)
    #test_op(2, 4, 128, 64, ws=3, NB=2, attn_bias=False, dtype=torch.float16)

    fc = '\n'.join([str(k)+': '+str(v)  for k, v in _attn_fwd.cache.items()])
    bc = '\n'.join([str(k) + ': ' + str(v) for k, v in _attn_bwd.cache.items()])
    pc = '\n'.join([str(k) + ': ' + str(v) for k, v in _attn_bwd_preprocess.cache.items()])
    print(f'attn_fwd cache:\n{fc}')
    print(f'attn_bwd_pre cache:\n{pc}')
    print(f'attn_bwd cache:\n{bc}')

    try:
        from flash_attn.flash_attn_interface import \
            flash_attn_qkvpacked_func as flash_attn_func
        HAS_FLASH = True
    except BaseException:
        HAS_FLASH = False

    BATCH, N_HEADS, N_CTX, D_HEAD = 4, 16, 256, 64
    kv_group = 1
    # vary seq length for fixed head and batch=4
    configs = []
    for BATCH in [2,4]:
        for mode in ["fwd", "bwd"]:
            for mask in [True]:
                for ws in [3,4,-4]:
                    #ws = 3
                    configs.append(
                        triton.testing.Benchmark(
                            x_names=["blk"],
                            x_vals=[2,4,8,12,16,24],
                            line_arg="provider",
                            line_vals=["triton"] + (["flash"] if HAS_FLASH else []),
                            line_names=["Triton"] + (["Flash-2"] if HAS_FLASH else []),
                            styles=[("red", "-"), ("blue", "-")],
                            ylabel="ms",
                            plot_name=
                            f"FA-h{N_HEADS}-b{BATCH}-d{D_HEAD}-ws{ws}-{mode}-mask{mask}-kvg{kv_group}.png",
                            args={
                                "H": N_HEADS,
                                "BATCH": BATCH,
                                "D_HEAD": D_HEAD,
                                "dtype": torch.float16,
                                "mode": mode,
                                'ws': ws,
                                'N_CTX': 64,
                                'mask': mask,
                                'kv_group':kv_group,
                            },
                        ))


    @triton.testing.perf_report(configs)
    def bench_flash_attention(BATCH, H, N_CTX, D_HEAD, mode, provider, ws,blk,mask, dtype=torch.float16,
                              kv_group = kv_group):
        assert mode in ["fwd", "bwd"]
        warmup = 15
        rep = 120
        device='cuda'
        if provider == "triton":
            NC = N_CTX
            q = torch.randn((BATCH, H,blk, blk, NC, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
            k = torch.randn((BATCH, H,blk, blk, NC, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
            v = torch.randn((BATCH, H,blk, blk, NC, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
            if mask:
                b = torch.randn((BATCH,H,blk,blk,NC,ws*ws*NC),dtype=dtype,device='cuda')
            else:
                b = None
            sm_scale = 0.5
            fn = lambda: block_attention(q, k, v, b, ws, ws, sm_scale)
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        elif provider == "flash":
            qkv = torch.randn((BATCH, N_CTX * blk * blk, 3, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)
            fn = lambda: flash_attn_func(qkv, causal=False)
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        else:
            raise NotImplementedError
        flops_per_matmul = 2.0 * BATCH * H * D_HEAD * (N_CTX ** 2)
        total_flops = 2 * flops_per_matmul
        if mode == "bwd":
            total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
        return ms

    # only works on post-Ampere GPUs right now
    bench_flash_attention.run(save_path=".", print_data=True)
