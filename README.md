# vision_fused_attention
localized/window self attention based on pytorch &amp; triton &amp; flash_attn v2

# API

```

output = block_attention(q, k, v, b=None, 
    window_size1=3, window_size2=3, sm_scale = 1.0, dropout = 0)
```
q,k,v are 6-D tensors with shape B,H,ht,wt,S,D 
(Batch_size, num_of_heads,
num of height blocks, num of width blocks, 
num of tokens in a single block, dim of each head). 
Note that num_of_heads of K/V should be divided by num_of_heads of Q for MQA.

b is attention bias with shape B,H,ht,wt,S,QS or B,H,ht,wt,S,S, 
where Q=window_size1 * window_size2.

window_size1 and window_size2 implies which nearby blocks provide K/V 
for each specific Q blocks along the ht or wt dim.
Let N = window_size:
- N = 1,3,5,7,9, ... means 
getting K/V from the nearest N blocks; 
- N = -2,-3,-4, ... means dividing the blocks into groups  
each containing (-N) blocks along the height/width dim, 
and getting K/V within each group;
- N = 2,4,6,8, ... means dividing the blocks into groups each containing 2 blocks,
and blocks in each group getting the same K/V from the nearest N blocks.

If dropout > 0, bias (1, H, ht, wt, 1, S) with values 0/-inf
will be added to the attention matrix.

# TODO
fused_attention2 loads Q/KV based on 
(block_id % window_size) in the inner loop, while fused_attention
loads Q/KV based on the relative position to KV/Q. 
Performance test of the two versions should be done.