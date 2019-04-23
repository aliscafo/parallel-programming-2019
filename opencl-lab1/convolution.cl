__kernel void convolution(__global float *a, __global float *b, __global float *c, int n, int m) {
  int id = get_global_id(0);

  int i = id / n, j = id % n;

  if (i >= n || j >= n) return;

  int hm = (m - 1) / 2;

  for (int k = -hm; k <= hm; k++)
    for (int l = -hm; l <= hm; l++)
      if (0 <= i + k && i + k < n && 0 <= j + l && j + l < n)
        c[i * n + j] += (double)a[(i + k) * n + (j + l)] * (double)b[(k + hm) * m + (l + hm)];

}