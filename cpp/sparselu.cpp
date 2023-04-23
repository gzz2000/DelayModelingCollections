// a sparse LU (actually LDL) decomposition implementation
// for R and C parasitics

#include <cstdio>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <algorithm>

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef intptr_t isize;
typedef uintptr_t usize;

const u32 u32_MAX = ~(u32)0;
// const usize usize_MAX = ~(usize)0;
// const u64 u64_MAX = ~(u64)0;

typedef float f32;
typedef double f64;

void debug_sparse_matrix(u32 n, const u32 nnz[], const u32 p[], const f32 v[]) {
	for(u32 i = 0; i < n; ++i) {
		printf("{%d}: ", i);
		for(u32 j = nnz[i]; j < nnz[i + 1]; ++j) {
			printf("[%d]=%f  ", p[j], v[j]);
		}
		putchar('\n');
	}
}

const usize NELEM = 55;
const usize N = 24;
const usize NMAT = NELEM * 2 + N;
const u32 Q = 4;

// solve linear equation (in-place) given the LDL decomposition.
// L is in CSC format w/o the diagonal 1s.
inline void lu_solve_(u32 n, f32 x[], const u32 Lnnz[], const u32 Lp[], const f32 Lv[], const f32 D[]) {
	// lower diagonal solve (L)
	for(u32 i = 0; i < n; ++i) {
		f32 xi = x[i];
		for(u32 jp = Lnnz[i], jpe = Lnnz[i + 1]; jp < jpe; ++jp) {
			x[Lp[jp]] -= Lv[jp] * xi;
		}
	}
	// diagonal solve (D)
	for(u32 i = 0; i < n; ++i) x[i] /= D[i];
	// upper diagonal solve (L^T)
	for(u32 i = n - 1; i != u32_MAX; --i) {
		f32 xi = x[i];
		for(u32 jp = Lnnz[i], jpe = Lnnz[i + 1]; jp < jpe; ++jp) {
			xi -= Lv[jp] * x[Lp[jp]];
		}
		x[i] = xi;
	}
}

// matrix-vector multiply.
// assumes y[] initialized as zero.
// given matrix is in CSC format.
inline void mvmul(u32 n, const f32 x[], const u32 nnz[], const u32 p[], const f32 v[], f32 y[]) {
	for(u32 i = 0; i < n; ++i) {
		f32 xi = x[i];
		for(u32 jp = nnz[i], jpe = nnz[i + 1]; jp < jpe; ++jp) {
			y[p[jp]] += v[jp] * xi;
		}
	}
}

inline f32 dot(u32 n, const f32 a[], const f32 b[]) {
	f32 r = 0.;
	for(u32 k = 0; k < n; ++k) r += a[k] * b[k];
	return r;
}

/// compute QR decomposition of Aq and store them in Qq and Rq.
/// this is non-traditional because the result is A=R*Q, not A=Q*R,
/// and R is lower-triangular, not upper-triangular.
/// assumes Aq is nonsingular.
/// If Aq is singular, it should be offsetted (shifted) by +lambda*I.
inline void QRdecQ(const f32 Aq[Q][Q], f32 Qq[Q][Q], f32 Rq[Q][Q]) {
	// init
	for(u32 i = 0; i < Q; ++i) {
		for(u32 j = 0; j < Q; ++j) {
			Qq[i][j] = Aq[i][j];
			Rq[i][j] = 0.;
		}
	}
	// Gram-Schmidt
	for(u32 i = 0; i < Q; ++i) {
		for(u32 j = 0; j < i; ++j) {
			f32 r = 0.;
			for(u32 k = 0; k < Q; ++k) r += Qq[j][k] * Aq[i][k];
			Rq[i][j] = r;
			for(u32 k = 0; k < Q; ++k) Qq[i][k] -= r * Qq[j][k];
		}
		// normalize current
		f32 norm = 0.;
		for(u32 k = 0; k < Q; ++k) norm += Qq[i][k] * Qq[i][k];
		norm = sqrt(norm);
		assert(abs(norm) > 1e-8);  // reject singularity
		Rq[i][i] = norm;
		for(u32 k = 0; k < Q; ++k) Qq[i][k] /= norm;
	}
}

// C = A @ B
// cannot be used in-place.
inline void matmulQ(const f32 A[Q][Q], const f32 B[Q][Q], f32 C[Q][Q]) {
	for(u32 i = 0; i < Q; ++i) for(u32 j = 0; j < Q; ++j) {
			f32 r = 0.;
			for(u32 k = 0; k < Q; ++k) r += A[i][k] * B[k][j];
			C[i][j] = r;
		}
}

typedef u8 ParasiticElementType;
constexpr ParasiticElementType ParasiticElementType_R = 0;
// constexpr ParasiticElementType ParasiticElementType_C = 1;

struct ParasiticElement {
	ParasiticElementType typ;
	u32 a, b;
	f32 value;
};

/// preprocess: only used during MNA construction. can be freed after that.
// sorting to obtain csc of system
u32 radixcnt[N + 1], radixorder1[NELEM], radixorder2[NELEM];
// general temp array
f32 diagG[N], diagC[N];
// temp counts
u32 Gfed[N], Cfed[N];

/// CSC representations of G and C.
// although they are symmetric, we store all elements.
// nnz arrays will be prefix-summed and become st arrays.
u32 Gnnz[N + 1], Gp[NMAT], Cnnz[N + 1], Cp[NMAT];
f32 Gv[NMAT], Cv[NMAT];

/// sparse LU helper arrays, can be freed after decomposition.
u32 parent[N], flag[N], Lfed[N], Lorder[N];
f32 Y[N];

/// sparse LU(LDL) result L matrix (in CSC) and D vector.
// the exact size of L is figured out only after symbolic decomposition.
u32 Lnnz[N + 1], Lp[NMAT * 2];
f32 Lv[NMAT * 2], D[N];

/// krylov subspace temporaries
/// Hq will be modified later during decomposition.
f32 Uq[Q + 1][N], Hq[Q][Q], Zq[Q][N];
/// evd temporaries
f32 Rq[Q][Q], Qq[Q][Q], Evecq[Q][Q];
/// ct-arnoldi result: poles and residues (should keep just
/// for the IO ports)
f32 poles[Q], residues[N][Q];

int main() {
	freopen("input.sparselu.txt", "r", stdin);
	// read in the matrices
	// (todo) we don't have sink cell caps here yet.
	u32 n = 0, n_elem;
	ParasiticElement elem[NELEM];
	
	scanf("%u", &n_elem);
	for(u32 i = 0; i < n_elem; ++i) {
		u32 petype;
		u32 a, b; f32 value;
		scanf("%u%u%u%f", &petype, &a, &b, &value);
		elem[i].typ = (u8)petype;
		elem[i].a = a; elem[i].b = b; elem[i].value = value;
	}
	f32 driver_rd;
	scanf("%f", &driver_rd);
	for(u32 i = 0; i < n_elem; ++i) {
		n = std::max(n, elem[i].a + 1);
		if(elem[i].b != u32_MAX) n = std::max(n, elem[i].b + 1);
	}

	// build matrix: preprocess -> csc
	// calculate nnz for G and C.
	for(u32 i = 1; i <= n; ++i) Gnnz[i] = Cnnz[i] = 1; // diag
	for(u32 i = 0; i < n_elem; ++i) {
		if(elem[i].typ == ParasiticElementType_R) {
			++Gnnz[elem[i].a + 1];
			++Gnnz[elem[i].b + 1];
		}
		else {
			if(elem[i].b == u32_MAX) continue;
			++Cnnz[elem[i].a + 1];
			++Cnnz[elem[i].b + 1];
		}
	}
	for(u32 i = 1; i <= n; ++i) {
		Gnnz[i] += Gnnz[i - 1];
		Cnnz[i] += Cnnz[i - 1];
	}
	// sort by: (b + 1, a + 1). (we already guaranteed a < b.)
	// +1 is for u32_MAX to become zero.
	// radix sort round 1: sort by a.
	for(u32 i = 0; i < n_elem; ++i) {
		u32 k = elem[i].a + 1;
		++radixcnt[k];
	}
	for(u32 i = 1; i <= n; ++i) radixcnt[i] += radixcnt[i - 1];
	for(u32 i = n_elem - 1; i != u32_MAX; --i) {
		u32 k = elem[i].a + 1;
		radixorder1[--radixcnt[k]] = i;
	}
	// radix sort round 2: sort by b.
	for(u32 i = 0; i <= n; ++i) radixcnt[i] = 0;
	for(u32 i = 0; i < n_elem; ++i) {
		u32 k = elem[i].b + 1;
		++radixcnt[k];
	}
	for(u32 i = 1; i <= n; ++i) radixcnt[i] += radixcnt[i - 1];
	for(u32 p = n_elem - 1; p != u32_MAX; --p) {
		u32 i = radixorder1[p];
		u32 k = elem[i].b + 1;
		radixorder2[--radixcnt[k]] = i;
	}
	// feed the upper triangular elements into G and C.
	// and store the diagonal element in tmp arrays
	diagG[0] = (f32)1. / driver_rd;
	for(u32 pos = 0; pos < n_elem; ++pos) {
		u32 i = radixorder2[pos];
		if(elem[i].typ == ParasiticElementType_R) {
			assert(elem[i].value > 1e-7);  // reject too small resistances?
			f32 invr = (f32)1. / elem[i].value;
			u32 a = elem[i].a, b = elem[i].b;
			diagG[a] += invr;
			diagG[b] += invr;
			u32 p = Gnnz[b] + Gfed[b]++;
			Gp[p] = a;
			Gv[p] = -invr;
		}
		else {
			f32 c = elem[i].value;
			u32 a = elem[i].a, b = elem[i].b;
			diagC[a] += c;
			if(b == u32_MAX) continue;
			diagC[b] += c;
			u32 p = Cnnz[b] + Cfed[b]++;
			Cp[p] = a;
			Cv[p] = -c;
		}
	}
	// append the diagonal elements
	for(u32 i = 0; i < n; ++i) {
		u32 p = Gnnz[i] + Gfed[i]++;
		Gp[p] = i;
		Gv[p] = diagG[i];
		p = Cnnz[i] + Cfed[i]++;
		Cp[p] = i;
		Cv[p] = diagC[i];
	}
	// append the transpose for these symmetric matrices
	for(u32 i = 0; i < n; ++i) {
		for(u32 j = 0, je = Gfed[i] - 1; j < je; ++j) {
			u32 jp = Gp[Gnnz[i] + j];
			f32 jv = Gv[Gnnz[i] + j];
			u32 p = Gnnz[jp] + Gfed[jp]++;
			Gp[p] = i;
			Gv[p] = jv;
		}
		for(u32 j = 0, je = Cfed[i] - 1; j < je; ++j) {
			u32 jp = Cp[Cnnz[i] + j];
			f32 jv = Cv[Cnnz[i] + j];
			u32 p = Cnnz[jp] + Cfed[jp]++;
			Cp[p] = i;
			Cv[p] = jv;
		}
	}
	// debug output the matrices
	// printf("G matrix = \n");
	// debug_sparse_matrix(n, Gnnz, Gp, Gv);
	// printf("C matrix = \n");
	// debug_sparse_matrix(n, Cnnz, Cp, Cv);

	// LU decomposition begins.
	// first: symbolic decomposition.
	for(u32 k = 0; k < n; ++k) {
		parent[k] = u32_MAX;
		flag[k] = k;
		for(u32 p = Gnnz[k], pe = Gnnz[k + 1]; p < pe; ++p) {
			u32 i = Gp[p];
			if(i >= k) break;
			for(; flag[i] != k; i = parent[i]) {
				++Lnnz[i + 1];   // column i has nonzero at row k.
				flag[i] = k;
				if(parent[i] == u32_MAX) parent[i] = k;
			}
		}
	}
	for(u32 i = 1; i <= n; ++i) Lnnz[i] += Lnnz[i - 1];
	assert(Lnnz[n] < NMAT * 2);  // our scratch code preallocates this.
	// next: numerical decomposition.
	for(u32 k = 0; k < n; ++k) {
		u32 top = n;
		flag[k] = k;
		for(u32 p = Gnnz[k], pe = Gnnz[k + 1]; p < pe; ++p) {
			u32 i = Gp[p];
			if(i > k) break;
			u32 len = 0;
			Y[i] += Gv[p];
			for(; flag[i] != k; i = parent[i]) {
				Lorder[len++] = i;
				flag[i] = k;
			}
			// reverse the inserted part
			while(len) Lorder[--top] = Lorder[--len];
		}
		// solve row k.
		D[k] = Y[k]; Y[k] = 0;
		while(top < n) {
			u32 i = Lorder[top++];
			f32 yi = Y[i]; Y[i] = 0;
			u32 p, pe;
			for(p = Lnnz[i], pe = Lnnz[i] + Lfed[i]; p < pe; ++p) {
				Y[Lp[p]] -= Lv[p] * yi;
			}
			f32 lki = yi / D[i];
			D[k] -= lki * yi;
			++Lfed[i];
			Lp[pe] = k;
			Lv[pe] = lki;
		}
		assert(fabs(D[k]) > 1e-8);  // reject singularity
	}
	
	// printf("L matrix = \n");
	// debug_sparse_matrix(n, Lnnz, Lp, Lv);
	// printf("D = ");
	// for(u32 i = 0; i < n; ++i) printf("%f  ", D[i]);
	// putchar('\n');

	// to check the result of printed L and D, run this:
	// >>> cL = np.linalg.cholesky(G)
	// >>> np.diag(cL) ** 2    # this should == D.
	// >>> (cL / np.diag(cL)).T    # this should == L excluding diagonal 1s.

	// ctarnoldi: krylov
	Uq[0][0] = (f32)1. / driver_rd;
	lu_solve_(n, Uq[0], Lnnz, Lp, Lv, D);
	mvmul(n, Uq[0], Cnnz, Cp, Cv, Zq[0]);
	f32 h00 = sqrt(dot(n, Uq[0], Zq[0]));  // also later used in residues
	for(u32 k = 0; k < n; ++k) Uq[0][k] /= h00, Zq[0][k] /= h00;
	for(u32 j = 1; j <= Q; ++j) {
		for(u32 k = 0; k < n; ++k) Uq[j][k] = -Zq[j - 1][k];
		lu_solve_(n, Uq[j], Lnnz, Lp, Lv, D);
		for(u32 i = (j <= 1 ? 0 : j - 2); i < j; ++i) {
			f32 t = dot(n, Uq[j], Zq[i]);
			Hq[i][j - 1] = t;
			for(u32 k = 0; k < n; ++k) Uq[j][k] -= t * Uq[i][k];
		}
		if(j >= Q) break;
		mvmul(n, Uq[j], Cnnz, Cp, Cv, Zq[j]);
		f32 hjpj = sqrt(dot(n, Uq[j], Zq[j]));
		Hq[j][j - 1] = hjpj;
		if(abs(hjpj) > 1e-5) {
			for(u32 k = 0; k < n; ++k) Uq[j][k] /= hjpj, Zq[j][k] /= hjpj;
		}
	}

	// test Uq, Hq
	puts("Hq = ");
	for(u32 i = 0; i < Q; ++i) {
		for(u32 j = 0; j < Q; ++j) printf("%f  ", Hq[i][j]);
		putchar('\n');
	}
	puts("Uq = ");
	for(u32 i = 0; i < Q; ++i) {
		for(u32 j = 0; j < n; ++j) printf("%f  ", Uq[i][j]);
		putchar('\n');
	}

	// eig factor Hq
	// first make Hq non-singular by shifting it to diagonally dominant.
	// also, we only subtract, not add, to keep some order in resulting eigs.
	f32 hmaxabs = 0., hmaxoffs = 0.;
	if(n < Q) {
		for(u32 i = 0; i < Q; ++i) for(u32 j = 0; j < Q; ++j) {
				hmaxabs = fmaxf(hmaxabs, abs(Hq[i][j]));
			}
		hmaxabs *= (f32)0.01;
		for(u32 i = 0; i < Q; ++i) {
			f32 nondiag = 0.;
			for(u32 j = 0; j < Q; ++j) if(i != j) nondiag += abs(Hq[i][j]);
			hmaxoffs = fmaxf(hmaxoffs, nondiag * (f32)1.01 + hmaxabs + Hq[i][i]);
		}
		printf("hmaxoffs to be: %f\n", hmaxoffs);
	}
	for(u32 i = 0; i < Q; ++i) {
		Hq[i][i] -= hmaxoffs;
		Evecq[i][i] = 1.;
	}
	// QR iterations
	for(u32 iter = 0; iter < 32; ++iter) {
		if(iter % 4 == 3) {
			// check convergence
			bool conv = true;
			for(u32 i = 0; i < Q && conv; ++i) for(u32 j = 0; j < Q; ++j) {
					if(i != j && abs(Hq[i][j]) > 1e-3) {
						conv = false;
						break;
					}
				}
			if(conv) {
				printf("QR converged in %d iters\n", iter);
				break;
			}
			else if(iter == 31) {
				printf("[WARNING] QR does not converge..\n");
			}
		}
		QRdecQ(Hq, Qq, Rq);
		matmulQ(Qq, Rq, Hq);
		matmulQ(Qq, Evecq, Rq);
		for(u32 i = 0; i < Q; ++i) for(u32 j = 0; j < Q; ++j) {
				Evecq[i][j] = Rq[i][j];
			}
	}
	for(u32 i = 0; i < Q; ++i) Hq[i][i] += hmaxoffs;

	// debug eig
	puts("eigenvalues = ");
	for(u32 i = 0; i < Q; ++i) printf("%f  ", Hq[i][i]);
	putchar('\n');
	puts("eigenvectors (rows) = ");
	for(u32 i = 0; i < Q; ++i) {
		for(u32 j = 0; j < Q; ++j) printf("%f  ", Evecq[i][j]);
		putchar('\n');
	}

	for(u32 i = 0; i < Q; ++i) {
		poles[i] = (f32)1. / Hq[i][i];
		for(u32 j = 0; j < n; ++j) {
			f32 r = 0.;
			for(u32 k = 0; k < Q; ++k) r += Uq[k][j] * Evecq[i][k];
			residues[j][i] = r * Evecq[i][0] * h00;
		}
	}
	// debug poles & res
	puts("poles = ");
	for(u32 i = 0; i < Q; ++i) printf("%f  ", poles[i]);
	putchar('\n');
	puts("residues = ");
	for(u32 i = 0; i < n; ++i) {
		for(u32 j = 0; j < Q; ++j) printf("%f  ", residues[i][j]);
		putchar('\n');
	}
}
