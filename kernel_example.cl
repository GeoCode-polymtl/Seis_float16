#define FDOH 2 //half of the stencil length

__device__ __inline__ half2 hp(half *a){ //Two consecutive halfs -> half2
    __prec2 output;
    *((half *)&output) = *a;
    *((half *)&output+1) = *(a+1);
    return output;
}

extern "C" __global__ void update_v(half2 *sxx, half2 *sxz, half2 *szz, //stress
                                    half2 *vx,  half2 *vz,   //velocity
                                    half2 *rip, half2 *rkp, //scaled buoyancies
                                    const int NX, // grid size in X
                                    const int NZ // half grid size in Z
                                    ){
    //Shared memory
    extern __shared__ half2 lvar2[];
    half * lvar=(half *)lvar2;

    //Grid position
    int lsizez = blockDim.x+2*FDOH/2;
    int lsizex = blockDim.y+2*FDOH;
    int lidz = threadIdx.x+FDOH/2;
    int lidx = threadIdx.y+FDOH;
    int gidz = blockIdx.x*blockDim.x+threadIdx.x+FDOH/2;
    int gidx = blockIdx.y*blockDim.y+threadIdx.y+FDOH;
    int indp = (gidx-FDOH)*(NZ-2*FDOH/2)+(gidz-FDOH/2);
    int indv = gidx*NZ+gidz;
    
    #define ind1(z,x)  (x)*2*lsizez+(z)
    #define ind2(z,x)  (x)*lsizez+(z)
    #define indg(z,x)  (x)*NZ+(z)

    //Calculation of the spatial derivatives for sxz
    lvar2[ind2(lidz,lidx)]=sxz[indg(gidz,gidx)] //load into shared memory
    if (lidx<2*FDOH) //load halo in x into shared memory
        lvar2[ind2(lidz,lidx-FDOH)]=sxz[indg(gidz,gidx-FDOH)];
    if (lidx>(lsizex-2*FDOH-1))
        lvar2[ind2(lidz,lidx+FDOH)]=sxz[indg(gidz,gidx+FDOH)];
    if (lidz<2*FDOH/2) //load halo in z into shared memory
        lvar2[ind2(lidz-FDOH/2,lidx)]=sxz[indg(gidz-FDOH/2,gidx)];
    if (lidz>(lsizez-2*FDOH/2-1))
        lvar2[ind2(lidz+FDOH/2,lidx)]=sxz[indg(gidz+FDOH/2,gidx)];
    __syncthreads();
    
    #define HC1  1.125
    #define HC2 -0.041666666666666664
    half2 sxz_x = (HC1*(lvar2[ind2(lidz,lidx)]   - lvar2[ind2(lidz,lidx-1)]) +
                   HC2*(lvar2[ind2(lidz,lidx+1)] - lvar2[ind2(lidz,lidx-2)]))
    //We must reorder half2 to have consecutive grid elements for FD stencil in z
    half2 sxz_z = (HC1*(hp(lvar[ind1(2*lidz,lidx)])   - hp(lvar[ind1(2*lidz-1,lidx)]))+
                   HC2*(hp(lvar[ind1(2*lidz+1,lidx)]) - hp(lvar[ind1(2*lidz-2,lidx)])))
    
    //Calculation of derivative of sxx and szz
        // same as sxz ...

    //Stop updating if we are outside of the grid
    if (gidz>(NZ-FDOH/2-1) ||  gidx>(NX-FDOH-1) ){
        return;
    }
   
    //Write updated values to global memory
    vx[indv] = vx[indv] + (sxx_x+sxz_z) * rip[indp];
    vz[indv] = vz[indv] + (szz_z+sxz_x) * rkp[indp];
}
