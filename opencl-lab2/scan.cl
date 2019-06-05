#define SWAP(a,b) {__local int * tmp=a; a=b; b=tmp;}

__kernel void scan_hillis_steele_per_block(__global float *input, __global float *output, int n, int depth, __local float *a, __local float *b) {
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint block_size = get_local_size(0);
    uint block_end_ind = (gid + 1) * depth - 1;

    if (block_end_ind < n) {
        a[lid] = b[lid] = input[block_end_ind];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint s = 1; s < block_size; s <<= 1) {
        if (lid > (s-1))
        {
            b[lid] = a[lid] + a[lid-s];
        }
        else
        {
            b[lid] = a[lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        SWAP(a,b);
    }

    if (block_end_ind < n) {
    	output[block_end_ind] = a[lid];
    }
}


__kernel void scan_hillis_steele_propagation(__global float *input, __global float *output, int n, int depth) {
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint block_size = get_local_size(0);
    uint block_end_ind = (gid + 1) * depth - 1;

    __local float cur_pref_sum;

    if (lid == 0 && block_end_ind >= depth * block_size) {
        cur_pref_sum = input[block_end_ind - (block_end_ind % (depth * block_size)) - 1];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if ((block_end_ind + 1) % (depth * block_size) == 0) {
    	output[block_end_ind] = input[block_end_ind];
    } 
    else {
    	output[block_end_ind] = input[block_end_ind] + cur_pref_sum;
    }
}
