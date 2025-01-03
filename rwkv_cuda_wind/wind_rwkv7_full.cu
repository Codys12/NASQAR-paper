#include <torch/extension.h>
#include <cuda_bf16.h>
#include <stdio.h>

using bf = __nv_bfloat16;

void cuda_forward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*s0, bf*y, bf*s, bf*sT);

void forward(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &z, torch::Tensor &a, torch::Tensor &s0, torch::Tensor &y, torch::Tensor &s, torch::Tensor &sT) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_forward(B, T, H, (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)z.data_ptr(), (bf*)a.data_ptr(), (bf*)s0.data_ptr(), (bf*)y.data_ptr(), (bf*)s.data_ptr(), (bf*)sT.data_ptr());
}

void cuda_backward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*dy, bf*s, bf*dsT, bf*dw, bf*dq, bf*dk, bf*dv, bf*dz, bf*da, bf*ds0);

void backward(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &z, torch::Tensor &a, torch::Tensor &dy,
        torch::Tensor &s, torch::Tensor &dsT, torch::Tensor &dw, torch::Tensor &dq, torch::Tensor &dk, torch::Tensor &dv, torch::Tensor &dz, torch::Tensor &da, torch::Tensor &ds0) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_backward(B, T, H, (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)z.data_ptr(), (bf*)a.data_ptr(), (bf*)dy.data_ptr(), 
            (bf*)s.data_ptr(), (bf*)dsT.data_ptr(), (bf*)dw.data_ptr(), (bf*)dq.data_ptr(), (bf*)dk.data_ptr(), (bf*)dv.data_ptr(), (bf*)dz.data_ptr(), (bf*)da.data_ptr(), (bf*)ds0.data_ptr());
}

/*TORCH_LIBRARY(wind, m) {
    m.def("forward", forward);
    m.def("backward", backward);
}*/

TORCH_LIBRARY(wind, m) {
    m.def("forward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor z, Tensor a, Tensor s0, Tensor(a!) y, Tensor(b!) s, Tensor(c!) sT) -> ()");
    m.def("backward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor z, Tensor a, Tensor dy, Tensor s, Tensor dsT, Tensor(a!) dw, Tensor(b!) dq, Tensor(c!) dk, Tensor(d!) dv, Tensor(e!) dz, Tensor(f!) da, Tensor(g!) ds0) -> ()");
}

TORCH_LIBRARY_IMPL(wind, CUDA, m) {
    m.impl("forward", &forward);
    m.impl("backward", &backward);
}

//TODO: static? inline? __align__(16)?

using bf = __nv_bfloat16;
using bf2 = __nv_bfloat162;
using uint = unsigned int;
__device__ inline float to_float(const bf & u) { return __bfloat162float(u); }
//__device__ inline bf to_bf(const float & u) { return 	__float2bfloat16_rn(u); }
__device__ inline bf to_bf(const float & u) {
float2 f2 = {u, 0.0f};
__hip_bfloat162 bf2 = __float22bfloat162_rn(f2);
return bf2.x;
}
__device__ inline float2 to_float2(const bf2 & u) { return 	__bfloat1622float2(u); }
__device__ inline float2 to_float2(const float2 & u) { return u; }
__device__ inline bf2 to_bf2(const float2 & u) { return __float22bfloat162_rn(u); }
__device__ inline uint& as_uint(const bf2&x) { return *((uint*)(&x)); }
__device__ inline uint __smem(const void*x) { return __cvta_generic_to_shared(x); }

__device__ void __commit_group() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ void __wait_group() { asm volatile("cp.async.wait_all;\n" ::); }
template<int N> __device__ void __wait_groups() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
    
__device__ void __copy_wait() { __commit_group(); __wait_group(); }

__device__ void operator*=(float2&a, const float2&b) { a.x *= b.x; a.y *= b.y; }
__device__ void operator+=(float2&a, const float2&b) { a.x += b.x; a.y += b.y; }
__device__ float2 operator+(const float2&a, const float2&b) { return {a.x+b.x,a.y+b.y}; }
__device__ float2 operator*(const float2&a, const float2&b) { return {a.x*b.x,a.y*b.y}; }

struct STile;
struct RTile;
struct FTile;

struct GTile {
    bf*ga;
    int stride;
    __device__ GTile(bf*ga_, int stride_) : ga(ga_), stride(stride_) {}
    __device__ GTile& operator=(const RTile&);
};
struct GFTile {
    float*ga;
    int stride;
    __device__ GFTile(float*ga_, int stride_) : ga(ga_), stride(stride_) {}
    __device__ GFTile& operator=(const FTile&);
};
struct STileT { STile*st; };

struct __align__(16) STile {
    bf data[16*16];
    __device__ STile() {}
    __device__ STile(const RTile&o) { *this=o; }
    __device__ STile& operator=(const GTile&);
    __device__ STile& operator=(const RTile&);
    __device__ STileT t() { return STileT{this}; }
};
struct Product { const RTile*a, *b; };
struct ProductPlus { const RTile*a, *b; const FTile* c; };
struct RTile {
    bf2 data[4];
    __device__ RTile() {}
    __device__ void zero_() { data[0] = data[1] = data[2] = data[3] = to_bf2({0.f,0.f}); }
    __device__ RTile(const STile&o) { *this=o; }
    __device__ RTile(const STileT&o) { *this=o; }
    __device__ RTile(const FTile&o) { *this=o; }
    __device__ RTile& operator=(const STile&);
    __device__ RTile& operator=(const STileT&);
    __device__ RTile& operator=(const FTile&fa);
    __device__ RTile& operator=(const GTile&);
};
struct FTile {
    union {
        float2 data[4];
        float fdata[8];
    };
    __device__ void zero_() { data[0] = data[1] = data[2] = data[3] = {0.f,0.f}; }
    __device__ FTile() {}
    __device__ FTile(const FTile&o) { for (int i = 0; i < 4; i++) data[i] = o.data[i]; }
    __device__ FTile(const RTile&r) { *this=r; }
    __device__ FTile(const Product&p) { *this=p; }
    __device__ FTile(const ProductPlus&p) { *this=p; }
    __device__ FTile& operator=(const Product&);
    __device__ FTile& operator=(const RTile&);
    __device__ FTile& operator=(const ProductPlus&);
    __device__ FTile& operator+=(const Product&);
    __device__ FTile& operator+=(const FTile&o) { for (int i = 0; i < 4; i++) data[i] += o.data[i]; return *this; }
};

__device__ void print(STile t) {
    if (threadIdx.x == 0) {
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                printf("%f ", to_float(t.data[i*16+j]));
            }
            printf("\n");
        }
        printf("\n");
    }
}

template<class T>
__device__ void print(T t, int warpi = 0) {
    int tid = threadIdx.x - warpi*32;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j += 2) {
            if (tid == i%8*4+j%8/2) {
                float2 xy = to_float2(t.data[i/8+j/8*2]);
                printf("%f %f ", xy.x, xy.y);
                //printf("T%d:{a%d,a%d} ", threadIdx.x, (i/8+j/8*2)*2, (i/8+j/8*2)*2+1);
            }
            __syncthreads();
        }
        if (tid == 0) printf("\n");
            __syncthreads();
    }
    if (tid == 0) printf("\n");
    __syncthreads();
}

template<class T>
__device__ void print8(T mat) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j += 2) {
            if (threadIdx.x == i%8*4+j%8/2) {
                float2 xy = to_float2(mat);
                printf("%f %f ", xy.x, xy.y);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) printf("\n");
            __syncthreads();
    }
    if (threadIdx.x == 0) printf("\n");
    __syncthreads();
}



__device__ void load(STile&sa, bf*ga, int stride) {
    int i = threadIdx.x%32/2, j = threadIdx.x%2;
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" :: "r"(__smem(&sa.data[i*16+j*8])), "l"(ga+stride*i+j*8), "n"(16));
}

__device__ void load(RTile&ra, const STile&sa) {
    int i = threadIdx.x%8, j = threadIdx.x%32/16, k = threadIdx.x/8%2;
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(as_uint(ra.data[0])), "=r"(as_uint(ra.data[1])), "=r"(as_uint(ra.data[2])), "=r"(as_uint(ra.data[3]))
            : "r"(__smem(&sa.data[i*16+j*8+k*8*16])));
}
__device__ void loadT(RTile&ra, const STile&sa) {
    int i = threadIdx.x%8, j = threadIdx.x%32/16, k = threadIdx.x/8%2;
    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(as_uint(ra.data[0])), "=r"(as_uint(ra.data[1])), "=r"(as_uint(ra.data[2])), "=r"(as_uint(ra.data[3]))
            : "r"(__smem(&sa.data[i*16+j*8*16+k*8])));
}

__device__ static inline void __m16n8k16(float2&d0, float2&d1, const bf2 &a0, const bf2 &a1, const bf2 &a2, const bf2 &a3, const bf2 &b0, const bf2 &b1, const float2 &c0, const float2 &c1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
            : "=f"(d0.x), "=f"(d0.y), "=f"(d1.x), "=f"(d1.y)
            : "r"(as_uint(a0)), "r"(as_uint(a1)), "r"(as_uint(a2)), "r"(as_uint(a3)),
              "r"(as_uint(b0)), "r"(as_uint(b1)),
              "f"(c0.x), "f"(c0.y), "f"(c1.x), "f"(c1.y));
}
__device__ void mma(FTile&rd, const RTile&ra, const RTile&rb, const FTile&rc) { // d = a*b^T + c
    __m16n8k16(rd.data[0],rd.data[1], ra.data[0],ra.data[1],ra.data[2],ra.data[3], rb.data[0],rb.data[2], rc.data[0],rc.data[1]);
    __m16n8k16(rd.data[2],rd.data[3], ra.data[0],ra.data[1],ra.data[2],ra.data[3], rb.data[1],rb.data[3], rc.data[2],rc.data[3]);
}
__device__ static inline void __m16n8k16(float2&d0, float2&d1, const bf2 &a0, const bf2 &a1, const bf2 &a2, const bf2 &a3, const bf2 &b0, const bf2 &b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
            : "+f"(d0.x), "+f"(d0.y), "+f"(d1.x), "+f"(d1.y)
            : "r"(as_uint(a0)), "r"(as_uint(a1)), "r"(as_uint(a2)), "r"(as_uint(a3)),
              "r"(as_uint(b0)), "r"(as_uint(b1)),
              "f"(d0.x), "f"(d0.y), "f"(d1.x), "f"(d1.y));
}
__device__ void mma(FTile&rd, const RTile&ra, const RTile&rb) { // d += a*b^T
    __m16n8k16(rd.data[0],rd.data[1], ra.data[0],ra.data[1],ra.data[2],ra.data[3], rb.data[0],rb.data[2]);
    __m16n8k16(rd.data[2],rd.data[3], ra.data[0],ra.data[1],ra.data[2],ra.data[3], rb.data[1],rb.data[3]);
}
__device__ void mm(FTile&rd, const RTile&ra, const RTile&rb) { // d = a*b^T
    __m16n8k16(rd.data[0],rd.data[1], ra.data[0],ra.data[1],ra.data[2],ra.data[3], rb.data[0],rb.data[2], {0.f,0.f}, {0.f,0.f});
    __m16n8k16(rd.data[2],rd.data[3], ra.data[0],ra.data[1],ra.data[2],ra.data[3], rb.data[1],rb.data[3], {0.f,0.f}, {0.f,0.f});
}

__device__ void store(const FTile&ra, float*ga, int stride) {
    int i = threadIdx.x%32/4, j = threadIdx.x%4*2;
    *((float2*)&ga[ i   *stride+j  ]) = ra.data[0];
    *((float2*)&ga[(i+8)*stride+j  ]) = ra.data[1];
    *((float2*)&ga[ i   *stride+j+8]) = ra.data[2];
    *((float2*)&ga[(i+8)*stride+j+8]) = ra.data[3];
}

__device__ void store(const RTile&ra, bf*ga, int stride) {
    int i = threadIdx.x%32/4, j = threadIdx.x%4*2;
    *((bf2*)&ga[ i   *stride+j  ]) = ra.data[0];
    *((bf2*)&ga[(i+8)*stride+j  ]) = ra.data[1];
    *((bf2*)&ga[ i   *stride+j+8]) = ra.data[2];
    *((bf2*)&ga[(i+8)*stride+j+8]) = ra.data[3];
}
__device__ void load(RTile&ra, bf*ga, int stride) {
    int i = threadIdx.x%32/4, j = threadIdx.x%4*2;
    ra.data[0] = *((bf2*)&ga[ i   *stride+j  ]);
    ra.data[1] = *((bf2*)&ga[(i+8)*stride+j  ]);
    ra.data[2] = *((bf2*)&ga[ i   *stride+j+8]);
    ra.data[3] = *((bf2*)&ga[(i+8)*stride+j+8]);
}
__device__ void store(const RTile&ra, STile&sa) { //TODO: reduce bank conflicts?
    int i = threadIdx.x%32/4, j = threadIdx.x%4*2;
    *((bf2*)&sa.data[ i   *16+j  ]) = ra.data[0];
    *((bf2*)&sa.data[(i+8)*16+j  ]) = ra.data[1];
    *((bf2*)&sa.data[ i   *16+j+8]) = ra.data[2];
    *((bf2*)&sa.data[(i+8)*16+j+8]) = ra.data[3];
}

__device__ void convert(RTile&ra, const FTile&fa) {
    ra.data[0] = to_bf2(fa.data[0]);
    ra.data[1] = to_bf2(fa.data[1]);
    ra.data[2] = to_bf2(fa.data[2]);
    ra.data[3] = to_bf2(fa.data[3]);
}
__device__ void convert(FTile&fa, const RTile&ra) {
    fa.data[0] = to_float2(ra.data[0]);
    fa.data[1] = to_float2(ra.data[1]);
    fa.data[2] = to_float2(ra.data[2]);
    fa.data[3] = to_float2(ra.data[3]);
}

__device__ STile& STile::operator=(const GTile& ga) { load(*this, ga.ga, ga.stride); return *this; }
__device__ RTile& RTile::operator=(const GTile& ga) { load(*this, ga.ga, ga.stride); return *this; }
__device__ RTile& RTile::operator=(const STile& sa) { load(*this, sa); return *this; }
__device__ STile& STile::operator=(const RTile& ra) { store(ra, *this); return *this; }
__device__ RTile& RTile::operator=(const STileT& sa) { loadT(*this, *sa.st); return *this; }
__device__ Product operator%(const RTile&ra, const RTile&rb) { return Product{&ra,&rb}; }
__device__ ProductPlus operator+(const Product&prod, const FTile&rc) { return ProductPlus{prod.a,prod.b,&rc}; }
__device__ FTile& FTile::operator=(const Product& prod) { mm(*this, *prod.a, *prod.b); return *this; }
__device__ FTile& FTile::operator=(const ProductPlus& prod) { mma(*this, *prod.a, *prod.b, *prod.c); return *this; }
__device__ FTile& FTile::operator+=(const Product& prod) { mma(*this, *prod.a, *prod.b); return *this; }
__device__ RTile& RTile::operator=(const FTile&fa) { convert(*this,fa); return *this; }
__device__ FTile& FTile::operator=(const RTile&ra) { convert(*this,ra); return *this; }
__device__ GTile& GTile::operator=(const RTile&ra) { store(ra, this->ga, this->stride); return *this; }
__device__ GFTile& GFTile::operator=(const FTile&fa) { store(fa, this->ga, this->stride); return *this; }

// Is this kind of cumsum better than multiplying with a triangular matrix of ones?
template<int inclusive, int rev>
__device__ FTile cumsumv(FTile&w) {
    int tid = threadIdx.x%32, t = tid/4;

    FTile ret;
    if (inclusive) for (int i = 0; i < 4; i++) ret.data[i] = w.data[i];
    else for (int i = 0; i < 4; i++) ret.data[i] = float2{0.f,0.f};

    for (int b = 0; b < 3; b++) {
        for (int i = 0; i < 8; i++) {
            float other_w = __shfl_xor_sync(0xffffffff, w.fdata[i], 4<<b);
            if ((t>>b)%2 == !rev) ret.fdata[i] += other_w;
            w.fdata[i] += other_w;
        }
    }
    for (int i : {0,1,4,5}) {
        float &w0 = w.fdata[i^(2*!rev)], &w1 = w.fdata[i^(2*rev)];
        ret.fdata[i^(2*!rev)] += w1;
        w0 += w1;
        w1 = w0;
    }
    return ret;
}

template<int inclusive, int rev>
__device__ FTile cumprodv(FTile&w) {
    int tid = threadIdx.x%32, t = tid/4;

    FTile ret;
    if (inclusive) for (int i = 0; i < 4; i++) ret.data[i] = w.data[i];
    else for (int i = 0; i < 4; i++) ret.data[i] = float2{1.f,1.f};

    for (int b = 0; b < 3; b++) {
        for (int i = 0; i < 8; i++) {
            float other_w = __shfl_xor_sync(0xffffffff, w.fdata[i], 4<<b);
            if ((t>>b)%2 == !rev) ret.fdata[i] *= other_w;
            w.fdata[i] *= other_w;
        }
    }
    for (int i : {0,1,4,5}) {
        float &w0 = w.fdata[i^(2*!rev)], &w1 = w.fdata[i^(2*rev)];
        ret.fdata[i^(2*!rev)] *= w1;
        w0 *= w1;
        w1 = w0;
    }
    return ret;
}

__device__ FTile operator*(const FTile&a, const FTile&b) {
    FTile ret;
    for (int i = 0; i < 8; i++) ret.fdata[i] = a.fdata[i]*b.fdata[i];
    return ret;
}

template<int triangular = 0, int WARPS> // Lower triangular
__device__ FTile sum_warp(float2*share, const FTile&f) {
    int tid = threadIdx.x%32, warpi = threadIdx.x/32;
    FTile sum;
    sum.zero_();
    for (int i : {0,1,2,3}) {
        if (i == 2 && triangular) continue;
        for (int j = 0; j < WARPS; j++) {
            if (warpi == j) share[tid] = f.data[i];
            __syncthreads();
           sum.data[i].x += share[tid].x;
           sum.data[i].y += share[tid].y;
            __syncthreads();
        }
    }
    return sum;
}

__device__ RTile from_warp(const RTile&ra, int src, float4*share) {
    int tid = threadIdx.x%32, warpi = threadIdx.x/32;
    RTile ret;
    if (warpi == src) share[tid] = *((float4*)ra.data);
    __syncthreads();
    *((float4*)ret.data) = share[tid];
    __syncthreads();
    return ret;
}

// inv(I-f) where f is strictly lower triangular
__device__ FTile tri_minv(const FTile&f, float*share) {
    int i0 = threadIdx.x%32/4, j0 = threadIdx.x%4*2;
    float inv[16] = {};
    for (int k = 0; k < 8; k++) {
        int i = i0+k/2%2*8, j = j0+k%2+k/4*8;
        share[i*16+j] = f.fdata[k];
    }
    int tid = threadIdx.x%32;
    inv[tid%16] = 1;
    for (int i = 1; i < 16; i++) {
        for (int j = 0; j < i; j++) {
            float fac = share[i*16+j];
            inv[i] += fac*inv[j];
        }
    }
    for (int i = 0; i < 16; i++)
        share[tid*16+i] = inv[i];
    FTile ret;
    for (int k = 0; k < 8; k++) {
        int i = i0+k/2%2*8, j = j0+k%2+k/4*8;
        ret.fdata[k] = share[j*16+i];
    }
    return ret;
}

template<int strict>
__device__ FTile tril(const FTile&f) {
    int i0 = threadIdx.x%32/4, j0 = threadIdx.x%4*2;
    FTile ret;
    for (int k = 0; k < 8; k++) {
        int i = i0+k/2%2*8, j = j0+k%2+k/4*8;
        if (strict) ret.fdata[k] = (i>j ? f.fdata[k] : 0.f);
        else ret.fdata[k] = (i>=j ? f.fdata[k] : 0.f);
    }
    return ret;
}

template<class F>
__device__ void apply_(FTile&tile, F f) {
    for (int i = 0; i < 8; i++) tile.fdata[i] = f(tile.fdata[i]);
}

__device__ bf2 transpose(bf2 a) {
    bf2 ret;
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(as_uint(ret)) : "r"(as_uint(a)));
    return ret;
}

__device__ RTile transpose(const RTile&ra) {
    RTile rb;
    rb.data[0] = transpose(ra.data[0]);
    rb.data[1] = transpose(ra.data[2]);
    rb.data[2] = transpose(ra.data[1]);
    rb.data[3] = transpose(ra.data[3]);
    return rb;
}

template<int strict>
__device__ FTile slow_dw(const RTile&A, const RTile&q, const RTile&k, STile*share) {
    share[0] = A;
    share[1] = q;
    share[2] = k;
    __syncthreads();
    if (threadIdx.x%32 == 0) {
        for (int k = 0; k < 16; k++) {
            for (int j = 0; j < 16; j++) {
                float sum = 0;
                for (int l = 0; l < k; l++) {
                    for (int r = k+strict; r < 16; r++) {
                        sum += to_float(share[0].data[r*16+l]) * to_float(share[1].data[r*16+j]) * to_float(share[2].data[l*16+j]);
                    }
                }
                share[3].data[k*16+j] = to_bf(sum);
            }
        }
    }
    __syncthreads();
    RTile ret = (RTile)share[3];
    __syncthreads();
    return ret;
}


__device__ static inline void __m16n8k8(float2&d0, float2&d1, const bf2 &a0, const bf2 &a1, const bf2 &b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
            : "=f"(d0.x), "=f"(d0.y), "=f"(d1.x), "=f"(d1.y) : "r"(as_uint(a0)), "r"(as_uint(a1)), "r"(as_uint(b0)), "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));
}

template<int strict>
__device__ RTile fast_dw(const RTile&A, const RTile&q, const RTile&k) {
    float2 qkA8[4];
    RTile kt = transpose(k), qt = transpose(q);
    __m16n8k8(qkA8[0],qkA8[1], qt.data[2], qt.data[3], transpose(A.data[1]));
    __m16n8k8(qkA8[2],qkA8[3], kt.data[0], kt.data[1], A.data[1]);
    for (int x : {0,1}) {
        qkA8[x] *= to_float2(kt.data[x]);
        qkA8[2+x] *= to_float2(qt.data[2+x]);
    }

    int tid = threadIdx.x%32, j = threadIdx.x%4;
    // Non-inclusive cumsum
    for (int i = 0; i < 4; i++) {
        float sum = qkA8[i].x+qkA8[i].y;
        float psum = __shfl_xor_sync(0xffffffff, sum, 1);
        float ppsum = __shfl_xor_sync(0xffffffff, sum+psum, 2);
        if (i < 2) {
            psum = ppsum*(j>=2)+psum*(j%2);
            qkA8[i].y = psum + qkA8[i].x;
            qkA8[i].x = psum;
        } else {
            psum = ppsum*(j<2)+psum*(j%2==0);
            qkA8[i].x = psum + qkA8[i].y;
            qkA8[i].y = psum;
        }
    }

    float2 qkA4[4];
    {
        RTile k_q;
        for (int i = 0; i < 8; i++) ((bf*)k_q.data)[i] = (j<2?((bf*)kt.data)[i]:((bf*)qt.data)[i]);
        float lower_left = (tid >= 16 && j < 2);
        bf2 A0 = to_bf2(to_float2(A.data[0])*float2{lower_left,lower_left});
        bf2 A3 = to_bf2(to_float2(A.data[3])*float2{lower_left,lower_left});
        __m16n8k8(qkA4[0],qkA4[1], k_q.data[0], k_q.data[1], A0 + transpose(A0));
        __m16n8k8(qkA4[2],qkA4[3], k_q.data[2], k_q.data[3], A3 + transpose(A3));
        for (int i = 0; i < 4; i++)
            qkA4[i] *= to_float2(k_q.data[i]);
    }

    // Non-inclusive cumsum
    for (int i = 0; i < 4; i++) {
        float sum = qkA4[i].x+qkA4[i].y;
        float psum = __shfl_xor_sync(0xffffffff, sum, 1);
        psum *= (j%2 == j<2);
        qkA4[i] = {psum + qkA4[i].y*(j>=2), psum + qkA4[i].x*(j<2)};
    }

    FTile ret;
    ret.data[0] = qkA8[0]+qkA4[0];
    ret.data[1] = qkA8[1]+qkA4[1];
    ret.data[2] = qkA8[2]+qkA4[2];
    ret.data[3] = qkA8[3]+qkA4[3];

    for (int ci : {0,1}) {
        for (int ti : {0,1}) {
            int Ai = ti*3, di = ti*2+ci;
            unsigned mask = 0xffff<<(j>=2)*16;
            bf A8x  = __shfl_sync(mask, A.data[Ai].x,  8+(j>=2)*18);
            bf A12x = __shfl_sync(mask, A.data[Ai].x, 12+(j>=2)*18);
            bf A12y = __shfl_sync(mask, A.data[Ai].y, 12+(j>=2)*18);
            bf2 nq = __shfl_xor_sync(0xffffffff, qt.data[di], 1);
            bf2 pk = __shfl_xor_sync(0xffffffff, kt.data[di], 1);

            bool even = (j%2==0);
            float ax = to_float(even?A8x:A12x), ay = to_float(even?A12x:A12y), c = to_float(even?kt.data[di].x:qt.data[di].y);
            float2 b = to_float2(j%2?pk:nq);
            float d = (ax*b.x+ay*b.y)*c;
            ret.data[di].y += even*d;
            ret.data[di].x +=!even*d;
        }
    }

    if (!strict) {
        // Do we really need tril<1>()?
        ret += (kt % tril<1>(A)) * qt;
    }
    return transpose(ret);
}

__device__ void debug_set(RTile&ra, int i, int j, float v) {
    if (threadIdx.x%32 == i%8*4+j%8/2) ((bf*)ra.data)[i/8*2+j/8*4+j%2] = to_bf(v);
}

constexpr int WARPS = _C_/16;
constexpr int fw_stages = 1, bw_stages = 1;

__global__ void forward_kernel(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, F_ s0_, bf* y_, bf* s_, bf* sT_) {
    constexpr int C = _C_, K = 16;
    int bi = blockIdx.y, hi = blockIdx.x;
    extern __shared__ char smem_[];
    char*smem = smem_;

    STile *sw_ = (STile*)smem; smem += sizeof(STile)*fw_stages*WARPS;
    STile *sq_ = (STile*)smem; smem += sizeof(STile)*fw_stages*WARPS;
    STile *sk_ = (STile*)smem; smem += sizeof(STile)*fw_stages*WARPS;
    STile *sv_ = (STile*)smem; smem += sizeof(STile)*fw_stages*WARPS;
    STile *sa_ = (STile*)smem; smem += sizeof(STile)*fw_stages*WARPS;
    STile *sb_ = (STile*)smem; smem += sizeof(STile)*fw_stages*WARPS;
    char*share = (char*)smem;

    int stride = H*C;
    int warpi = threadIdx.x/32;

    auto push = [&](int t) {
        int off = bi*T*H*C + t*K*H*C + hi*C + warpi*16;
        int si = t%fw_stages;
        sw_[si*WARPS+warpi] = GTile(w_+off, stride);
        sq_[si*WARPS+warpi] = GTile(q_+off, stride);
        sk_[si*WARPS+warpi] = GTile(k_+off, stride);
        sv_[si*WARPS+warpi] = GTile(v_+off, stride);
        sa_[si*WARPS+warpi] = GTile(a_+off, stride);
        sb_[si*WARPS+warpi] = GTile(b_+off, stride);
    };
    for (int t = 0; t < fw_stages-1 && t < T/K; t++) push(t), __commit_group();

    FTile state[WARPS];
    for (int i = 0; i < WARPS; i++) {
        int off = bi*H*C*C + hi*C*C + warpi*16*C + i*16;
        RTile tmp;
        tmp = GTile(s0_+off, C);
        state[i] = tmp;
    }

    for (int t = 0; t < T/K; t++) {
        __syncthreads();
        if (t+fw_stages-1 < T/K)
            push(t+fw_stages-1);
        __commit_group();
        __wait_groups<fw_stages-1>();
        __syncthreads();
        int si = t%fw_stages;
        STile &sw = sw_[si*WARPS+warpi], &sq = sq_[si*WARPS+warpi], &sk = sk_[si*WARPS+warpi], &sv = sv_[si*WARPS+warpi], &sa = sa_[si*WARPS+warpi], &sb = sb_[si*WARPS+warpi];

        FTile w = (RTile)sw;
        apply_(w, [](float x) { return __expf(-__expf(x)); });
        FTile fw = w;
        FTile non_incl_pref = cumprodv<0,0>(fw);
        FTile incl_pref = non_incl_pref * w;
        FTile inv_incl_pref = incl_pref;
        apply_(inv_incl_pref, [](float x) { return 1.f/x; });

        RTile wq = (RTile)sq *     incl_pref, kwi = (RTile)sk * inv_incl_pref;
        RTile wa = (RTile)sa * non_incl_pref, bwi = (RTile)sb * inv_incl_pref;
        FTile ab = sum_warp<1,WARPS>((float2*)share, tril<1>(wa % bwi));
        RTile ak = sum_warp<1,WARPS>((float2*)share, tril<1>(wa % kwi));

        RTile ab_inv;
        __syncthreads();
        if (threadIdx.x < 32) ab_inv = tri_minv(ab, (float*)share);
        __syncthreads();
        ab_inv = from_warp(ab_inv, 0, (float4*)share);

        RTile vt = sv.t();
        FTile ab_ut = vt % ak;
        for (int i = 0; i < WARPS; i++)
            ab_ut += state[i] % from_warp(wa, i, (float4*)share);
        RTile ut = FTile(ab_ut % ab_inv);

        FTile y = sum_warp<1,WARPS>((float2*)share, tril<0>(wq % kwi)) % vt;
        y +=      sum_warp<1,WARPS>((float2*)share, tril<0>(wq % bwi)) % ut;
        for (int i = 0; i < WARPS; i++)
            y += from_warp(wq, i, (float4*)share) % state[i];

        int off = bi*T*H*C + t*K*H*C + hi*C + warpi*16;
        GTile(y_+off, stride) = RTile(y);

        RTile kwt = transpose(kwi*fw), bwt = transpose(bwi*fw);
        for (int i = 0; i < WARPS; i++) {
            int off = bi*H*(T/K)*C*C + hi*(T/K)*C*C + t*C*C + warpi*16*C + i*16;
            GTile(s_+off, C) = (RTile)state[i];

            FTile fstate = state[i] * from_warp(fw, i, (float4*)share);
            fstate += vt % from_warp(kwt, i, (float4*)share);
            fstate += ut % from_warp(bwt, i, (float4*)share);
            state[i] = fstate;
        }
    }
    for (int i = 0; i < WARPS; i++) {
        int off = bi*H*C*C + hi*C*C + warpi*16*C + i*16;
        GTile(sT_+off, C) = state[i];
    }
}

void cuda_forward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*s0, bf*y, bf*s, bf*sT) {
    assert(T%16 == 0);
    constexpr int tmp_size1 = sizeof(float4)*32, tmp_size2 = sizeof(float)*16*16*2;
    constexpr int threads = 32*WARPS, shared_mem = sizeof(STile)*fw_stages*WARPS*6 + (tmp_size1 > tmp_size2 ? tmp_size1 : tmp_size2);
    static int reported = 0;
    if (!reported++) {
#if defined VERBOSE
        printf("forward_kernel() uses %d bytes of (dynamic) shared memory\n", shared_mem);
#endif
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, forward_kernel);
        int cur_mem = attr.maxDynamicSharedSizeBytes;
        if (shared_mem > cur_mem) {
#if defined VERBOSE
            printf("Increasing forward_kernel's MaxDynamicSharedMemorySize from %d to %d\n", cur_mem, shared_mem);
#endif
            assert(!cudaFuncSetAttribute(forward_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem));
        }
    }
    forward_kernel<<<dim3(H,B), dim3(threads), shared_mem>>>(T,H,w,q,k,v,z,a,s0,y,s,sT);
}


__global__ void backward_kernel(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, F_ dy_, F_ s_, F_ dsT_, bf* dw_, bf* dq_, bf* dk_, bf* dv_, bf* da_, bf* db_, bf* ds0_) {
    constexpr int C = _C_, K = 16;
    int bi = blockIdx.y, hi = blockIdx.x;
    extern __shared__ char smem_[];
    char*smem = smem_;

    STile *sw_ = (STile*)smem; smem += sizeof(STile)*bw_stages*WARPS;
    STile *sq_ = (STile*)smem; smem += sizeof(STile)*bw_stages*WARPS;
    STile *sk_ = (STile*)smem; smem += sizeof(STile)*bw_stages*WARPS;
    STile *sv_ = (STile*)smem; smem += sizeof(STile)*bw_stages*WARPS;
    STile *sa_ = (STile*)smem; smem += sizeof(STile)*bw_stages*WARPS;
    STile *sb_ = (STile*)smem; smem += sizeof(STile)*bw_stages*WARPS;
    STile *sdy_ = (STile*)smem; smem += sizeof(STile)*bw_stages*WARPS;
    STile *state_ = (STile*)smem; smem += sizeof(STile)*bw_stages*WARPS*WARPS;
    char*share = (char*)smem;

    int stride = H*C;
    int warpi = threadIdx.x/32;

    auto push = [&](int t) {
        int off = bi*T*H*C + t*K*H*C + hi*C + warpi*16;
        int si = t%fw_stages;
        sw_[si*WARPS+warpi] = GTile(w_+off, stride);
        sq_[si*WARPS+warpi] = GTile(q_+off, stride);
        sk_[si*WARPS+warpi] = GTile(k_+off, stride);
        sv_[si*WARPS+warpi] = GTile(v_+off, stride);
        sa_[si*WARPS+warpi] = GTile(a_+off, stride);
        sb_[si*WARPS+warpi] = GTile(b_+off, stride);
        sdy_[si*WARPS+warpi] = GTile(dy_+off, stride);
        for (int i = 0; i < WARPS; i++) {
            int off2 = bi*H*(T/K)*C*C + hi*(T/K)*C*C + t*C*C + warpi*16*C + i*16;
            state_[si*WARPS*WARPS+warpi*WARPS+i] = GTile(s_+off2, C);
        }
    };

    FTile dstate[WARPS];
    for (int i = 0; i < WARPS; i++) {
        int off = bi*H*C*C + hi*C*C + warpi*16*C + i*16;
        RTile tmp;
        tmp = GTile(dsT_+off, C);
        dstate[i] = tmp;
        __commit_group();
    }

    for (int t = 0; t < bw_stages-1 && t < T/K; t++) push(T/K-1-t), __commit_group();

    for (int t = T/K-1; t >= 0; t--) {
        __syncthreads();
        if (t-bw_stages+1 >= 0)
            push(t-bw_stages+1);
        __commit_group();
        __wait_groups<bw_stages-1>();
        __syncthreads();
        int si = t%bw_stages;
        STile &sw = sw_[si*WARPS+warpi], &sq = sq_[si*WARPS+warpi], &sk = sk_[si*WARPS+warpi], &sv = sv_[si*WARPS+warpi], &sa = sa_[si*WARPS+warpi], &sb = sb_[si*WARPS+warpi], &sdy = sdy_[si*WARPS+warpi];
        STile*state = state_+si*WARPS*WARPS;

        FTile w = (RTile)sw;
        apply_(w, [](float x) { return __expf(-__expf(x)); });
        FTile fw = w;
        FTile non_incl_pref = cumprodv<0,0>(fw);
        FTile incl_pref = non_incl_pref * w;
        FTile inv_incl_pref = incl_pref;
        apply_(inv_incl_pref, [](float x) { return 1.f/x; });

        RTile wq = (RTile)sq *     incl_pref, kwi = (RTile)sk * inv_incl_pref;
        RTile wa = (RTile)sa * non_incl_pref, bwi = (RTile)sb * inv_incl_pref;
        FTile ab = sum_warp<1,WARPS>((float2*)share, tril<1>(wa % bwi));
        RTile ak = sum_warp<1,WARPS>((float2*)share, tril<1>(wa % kwi));

        RTile ab_inv;
        __syncthreads();
        if (threadIdx.x < 32) ab_inv = tri_minv(ab, (float*)share);
        __syncthreads();
        ab_inv = from_warp(ab_inv, 0, (float4*)share);

        RTile vt = sv.t();
        FTile ab_ut = vt % ak;
        for (int i = 0; i < WARPS; i++)
            ab_ut += state[warpi*WARPS+i] % from_warp(wa, i, (float4*)share);
        RTile ut = FTile(ab_ut % ab_inv);

        RTile qb = sum_warp<1,WARPS>((float2*)share, tril<0>(wq % bwi));
        RTile qk = sum_warp<1,WARPS>((float2*)share, tril<0>(wq % kwi));

        RTile dyt = sdy.t();
        FTile dut = FTile(dyt % transpose(qb));
        FTile dv = transpose(qk) % dyt;
        for (int i = 0; i < WARPS; i++) {
            RTile dstatei = dstate[i];
            dut += dstatei % from_warp(bwi*fw, i, (float4*)share);
            dv += from_warp(kwi*fw, i, (float4*)share) % dstatei;
        }
        RTile dab_ut = FTile(dut % transpose(ab_inv));
        dv += transpose(ak) % dab_ut;

        int off = bi*T*H*C + t*K*H*C + hi*C + warpi*16;
        GTile(dv_+off, stride) = RTile(dv);

        FTile dab = sum_warp<1,WARPS>((float2*)share, tril<1>(transpose(dab_ut) % transpose(ut)));
        FTile dak = sum_warp<1,WARPS>((float2*)share, tril<1>(transpose(dab_ut) % transpose(vt)));
        FTile dab_u_state0;
        dab_u_state0.zero_();
        for (int i = 0; i < WARPS; i++)
            dab_u_state0 += from_warp(transpose(dab_ut), i, (float4*)share) % state[i*WARPS+warpi].t();

        FTile da = dab_u_state0;
        da += dab % transpose(bwi);
        da += dak % transpose(kwi);
        da = non_incl_pref * da;
        GTile(da_+off, stride) = RTile(da);

        FTile dqb = sum_warp<1,WARPS>((float2*)share, tril<0>(transpose(dyt) % transpose(ut)));
        FTile dqk = sum_warp<1,WARPS>((float2*)share, tril<0>(transpose(dyt) % transpose(vt)));
        FTile dy_state0;
        dy_state0.zero_();
        for (int i = 0; i < WARPS; i++)
            dy_state0 += from_warp(transpose(dyt), i, (float4*)share) % state[i*WARPS+warpi].t();

        FTile dq = dy_state0;
        dq += dqb % transpose(bwi);
        dq += dqk % transpose(kwi);
        dq = incl_pref * dq;
        GTile(dq_+off, stride) = RTile(dq);

        RTile wqt = transpose(wq), wat = transpose(wa);

        FTile u_dstate, v_dstate, dw;
        u_dstate.zero_();
        v_dstate.zero_();
        dw.zero_();
        RTile ones;
        for (int i = 0; i < 4; i++) ones.data[i] = to_bf2({1.f,1.f});
        for (int i = 0; i < WARPS; i++) {
            int tid = threadIdx.x%32;
            if (warpi == i) {
                for (int j = 0; j < WARPS; j++) {
                    RTile ra = dstate[j];
                    ((float4*)share)[j*32+tid] = *((float4*)ra.data);
                }
            }
            RTile dstatei;// = dstate[i*WARPS+warpi];
            __syncthreads();
            *((float4*)dstatei.data) = ((float4*)share)[warpi*32+tid];
            __syncthreads();
            RTile dstatei_t = transpose(dstatei);
            v_dstate += from_warp(transpose(vt), i, (float4*)share) % dstatei_t;
            u_dstate += from_warp(transpose(ut), i, (float4*)share) % dstatei_t;
            dw += ones % transpose((RTile)state[i*WARPS+warpi]*dstatei);
        }

        FTile db = fw * u_dstate;
        db += transpose(dab) % wat;
        db += transpose(dqb) % wqt;
        db = inv_incl_pref * db;
        GTile(db_+off, stride) = RTile(db);

        FTile dk = fw * v_dstate;
        dk += transpose(dak) % wat;
        dk += transpose(dqk) % wqt;
        dk = inv_incl_pref * dk;
        GTile(dk_+off, stride) = RTile(dk);

        dw = fw * dw;
        dw += fast_dw<1>(dab,wa,bwi);
        dw += fast_dw<1>(dak,wa,kwi);
        dw += fast_dw<0>(dqb,wq,bwi);
        dw += fast_dw<0>(dqk,wq,kwi);
        FTile tmp;
        dw += cumsumv<0,0>(tmp = v_dstate*(fw*kwi));
        dw += cumsumv<0,0>(tmp = u_dstate*(fw*bwi));
        dw += cumsumv<0,1>(tmp = dab_u_state0*wa);
        dw += cumsumv<1,1>(tmp = dy_state0*wq);

        FTile dw_fac = (RTile)sw;
        apply_(dw_fac, [](float x) { return -__expf(x); });
        dw = dw * dw_fac;
        GTile(dw_+off, stride) = RTile(dw);

        __syncthreads();
        for (int i = 0; i < WARPS; i++) {
            FTile ndstate = dstate[i] * from_warp(fw, i, (float4*)share);
            ndstate += dyt % from_warp(wqt, i, (float4*)share);
            ndstate += dab_ut % from_warp(wat, i, (float4*)share);
            dstate[i] = ndstate;
        }
        __syncthreads();
    }
    for (int i = 0; i < WARPS; i++) {
        int off = bi*H*C*C + hi*C*C + warpi*16*C + i*16;
        GTile(ds0_+off, C) = dstate[i];
    }
}

void cuda_backward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*dy, bf*s, bf*dsT, bf*dw, bf*dq, bf*dk, bf*dv, bf*dz, bf*da, bf*ds0) {
    assert(T%16 == 0);
    constexpr int tmp_size1 = sizeof(float4)*32*WARPS, tmp_size2 = sizeof(float)*16*16*2;
    constexpr int threads = 32*WARPS, shared_mem = sizeof(STile)*WARPS*bw_stages*(7+WARPS) + (tmp_size1 > tmp_size2 ? tmp_size1 : tmp_size2);
    static int reported = 0;
    if (!reported++) {
#if defined VERBOSE
        printf("backward_kernel() uses %d bytes of (dynamic) shared memory\n", shared_mem);
#endif
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, backward_kernel);
        int cur_mem = attr.maxDynamicSharedSizeBytes;
        if (shared_mem > cur_mem) {
#if defined VERBOSE
            printf("Increasing backward_kernel's MaxDynamicSharedMemorySize from %d to %d\n", cur_mem, shared_mem);
#endif
            assert(!cudaFuncSetAttribute(backward_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem));
        }
    }
    backward_kernel<<<dim3(H,B), dim3(threads), shared_mem>>>(T,H,w,q,k,v,z,a,dy,s,dsT,dw,dq,dk,dv,dz,da,ds0);
}
