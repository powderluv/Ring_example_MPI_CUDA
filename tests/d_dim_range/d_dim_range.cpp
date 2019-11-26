//
// Created by Wei, Weile on 11/26/19.
//
#include <vector>
#include <assert.h>
#include <iostream>

double func(int i, int j, int k)
{
    return 42.0;
}

int main()
{
    std::vector<double> A_(120, 0);

    int istart=2, iend = 4, jstart=1, jend = 4, kstart = -3, kend = 6;
    int const isize = iend - istart + 1;
    int const jsize = jend - jstart + 1;

    // Note there are  4-2+1 = 3 values in 1st index,  4-1+1 = 4 values in 2nd index
    // 6 - (-3) + 1 = 10 values in 3rd index. Thus there are a total of 3*4*10 = 120 entries.

    auto A = [&] (int const i, int const j, int const k) -> double & {
        // note  return type -> T& or use -> T& const for read only array
        assert( (istart <= i) && (i <= iend));
        assert( (jstart <= j) && (j <= jend));
        assert( (kstart <= k) && (k <= kend));

        return A_[(i - istart) + (j - jstart) * isize + (k - kstart) *(isize * jsize)] ;
    };

    // we can then use this expression directly as
    for(int k=kstart; k <= kend; k++) {
        for(int j=jstart; j <= jend; j++) {
            for( int i=istart; i <= iend; i++) {
                A(i,j,k) = func(i,j,k);
            };
        };
    };
}