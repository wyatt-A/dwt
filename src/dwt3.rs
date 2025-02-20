use crate::wavelet::{Wavelet, WaveletFilter, WaveletType};
use ndarray::{Array3, ShapeBuilder};
use num_complex::Complex32;
use std::time::Instant;

#[test]
fn test_dwt3_axis() {
    let mut x = Array3::<Complex32>::from_shape_fn((120, 120, 120).f(), |(i, _, _)| Complex32::new(i as f32, 0.0));


    let w: Wavelet<f32> = Wavelet::new(WaveletType::Daubechies2);

    let n_coeffs = (120 + w.filt_len() - 1) / 2;

    let mut r = Array3::<Complex32>::zeros((2 * n_coeffs, 120, 120).f());

    let now = Instant::now();
    dwt3_axis(
        x.as_slice_memory_order().unwrap(),
        r.as_slice_memory_order_mut().unwrap(),
        &[120, 120, 120],
        0,
        &w,
    );
    let elapsed = now.elapsed();
    println!("xform took {} ms", elapsed.as_millis());
    //cfl::dump_magnitude("x", &r.into_dyn());

    idwt3_axis(
        r.as_slice_memory_order().unwrap(),
        x.as_slice_memory_order_mut().unwrap(),
        &[120, 120, 120],
        &[122, 120, 120],
        0,
        &w,
    );

    cfl::dump_magnitude("x", &x.into_dyn());
}

#[test]
fn test_dwt3_all() {
    let input_size = [512, 256, 128];
    let mut x = Array3::<Complex32>::from_shape_fn(input_size.f(), |(i, j, k)| Complex32::new((i + j + k) as f32, 0.0));

    let now = Instant::now();
    let w: Wavelet<f32> = Wavelet::new(WaveletType::Daubechies2);
    let result_size = result_size(&input_size, w.filt_len());
    let tmp1_size = [result_size[0], input_size[1], input_size[2]];
    let tmp2_size = [result_size[0], result_size[1], input_size[2]];

    let mut tmp1 = Array3::zeros(tmp1_size.f());
    let mut tmp2 = Array3::zeros(tmp2_size.f());
    let mut result = Array3::zeros(result_size.f());

    dwt3_axis(
        x.as_slice_memory_order().unwrap(),
        tmp1.as_slice_memory_order_mut().unwrap(),
        &input_size,
        0,
        &w,
    );

    dwt3_axis(
        tmp1.as_slice_memory_order().unwrap(),
        tmp2.as_slice_memory_order_mut().unwrap(),
        &tmp1_size,
        1,
        &w,
    );

    dwt3_axis(
        tmp2.as_slice_memory_order().unwrap(),
        result.as_slice_memory_order_mut().unwrap(),
        &tmp2_size,
        2,
        &w,
    );


    idwt3_axis(
        result.as_slice_memory_order().unwrap(),
        tmp2.as_slice_memory_order_mut().unwrap(),
        &tmp2_size,
        &result_size,
        2,
        &w,
    );

    idwt3_axis(
        tmp2.as_slice_memory_order().unwrap(),
        tmp1.as_slice_memory_order_mut().unwrap(),
        &tmp1_size,
        &tmp2_size,
        1,
        &w,
    );

    idwt3_axis(
        tmp1.as_slice_memory_order().unwrap(),
        x.as_slice_memory_order_mut().unwrap(),
        &input_size,
        &tmp1_size,
        0,
        &w,
    );

    let elapsed = now.elapsed();

    println!("3D wavelet decomp took {} ms", elapsed.as_millis());

    cfl::dump_magnitude("x", &x.into_dyn());
}

fn result_size(vol_size: &[usize; 3], f_len: usize) -> [usize; 3] {
    let mut r = [0, 0, 0];
    vol_size.iter().zip(r.iter_mut()).for_each(|(&d, r)| {
        let tmp = (d + f_len - 1) / 2;
        *r = tmp * 2;
    });
    r
}

fn dwt3_axis(vol_data: &[Complex32], result: &mut [Complex32], vol_size: &[usize; 3], axis: usize, wavelet: &Wavelet<f32>) {
    assert!(axis < 3, "axis out of bounds");

    let lo_d = wavelet.lo_d();
    let hi_d = wavelet.hi_d();

    let f_len = wavelet.filt_len() as i32;
    let sig_len = vol_size[axis] as i32;

    // this is the jump to get from one signal element to the next across an axis
    let signal_stride = if axis == 0 {
        1
    } else if axis == 1 {
        vol_size[0]
    } else {
        vol_size[0] * vol_size[1]
    };

    // this is the stride to get from one axis lane to the next
    let (_, n_lanes) = if axis == 0 {
        (vol_size[0], vol_size[1] * vol_size[2])
    } else if axis == 1 {
        (1, vol_size[0] * vol_size[2])
    } else {
        (vol_size[0] * vol_size[1], vol_size[0] * vol_size[1])
    };

    let n_coeffs = (sig_len + f_len - 1) / 2;
    let ext_len = f_len as i32 - 1;

    let result_size = if axis == 0 {
        [2 * n_coeffs as usize, vol_size[1], vol_size[2]]
    } else if axis == 1 {
        [vol_size[0], 2 * n_coeffs as usize, vol_size[2]]
    } else {
        [vol_size[0], vol_size[1], 2 * n_coeffs as usize]
    };

    let result_stride = if axis == 0 {
        1
    } else if axis == 1 {
        vol_size[0]
    } else {
        vol_size[0] * vol_size[1]
    };

    for lane in 0..n_lanes {
        let signal_lane_head = lane_head(lane, axis, vol_size);
        let result_lane_head = lane_head(lane, axis, &result_size);
        for i in 0..n_coeffs as i32 {
            let mut sa = Complex32::ZERO;
            let mut sd = Complex32::ZERO;
            for j in 0..f_len as i32 {
                // calculate virtual signal index that may extend out-of-bounds
                let virtual_idx = 2 * i - ext_len + j + 1;
                // rationalize the index by imposing the boundary condition
                //let signal_idx = symmetric_boundary_index(virtual_idx, sig_len);

                let signal_idx = if virtual_idx >= 0 && virtual_idx < sig_len {
                    virtual_idx as usize
                } else if virtual_idx < 0 {
                    (virtual_idx + 1).abs() as usize
                } else {
                    (2 * sig_len - virtual_idx - 1) as usize
                };

                // calculate the filter index
                let filter_idx = (f_len - j - 1) as usize;
                // multiply signal with filter coefficient

                let actual_index = signal_idx * signal_stride + signal_lane_head;

                sa += lo_d[filter_idx] * vol_data[actual_index];
                sd += hi_d[filter_idx] * vol_data[actual_index];
            }

            let result_index_a = i as usize * result_stride + result_lane_head;
            let result_index_d = (i + n_coeffs) as usize * result_stride + result_lane_head;

            result[result_index_a] = sa;
            result[result_index_d] = sd;
        }
    }
}

fn idwt3_axis(decomp: &[Complex32], vol: &mut [Complex32], vol_size: &[usize; 3], decomp_size: &[usize; 3], axis: usize, wavelet: &Wavelet<f32>) {
    let decomp_stride = if axis == 0 {
        1
    } else if axis == 1 {
        decomp_size[0]
    } else {
        decomp_size[0] * decomp_size[1]
    };

    let signal_stride = if axis == 0 {
        1
    } else if axis == 1 {
        vol_size[0]
    } else {
        vol_size[0] * vol_size[1]
    };

    let n_lanes = if axis == 0 {
        decomp_size[1] * decomp_size[2]
    } else if axis == 1 {
        decomp_size[0] * decomp_size[2]
    } else {
        decomp_size[0] * decomp_size[1]
    };

    let n_coeffs = decomp_size[axis] as i32 / 2;
    let f_len = wavelet.filt_len() as i32;

    let lo_r = wavelet.lo_r();
    let hi_r = wavelet.hi_r();

    for lane in 0..n_lanes {
        let decomp_lane_head = lane_head(lane, axis, decomp_size);
        let signal_lane_head = lane_head(lane, axis, vol_size);

        //let approx = &result[0..n_coeffs as usize];
        //let detail = &result[n_coeffs as usize..];

        //let n = approx.len();
        //let m = lo_r.len();
        let full_len = 2 * n_coeffs + f_len - 1;        // length of full convolution
        let keep_len = 2 * n_coeffs - f_len + 2;       // how many samples we keep (centered)
        let start = (full_len - keep_len) / 2;

        //let mut recon = vec![0.; sig_len as usize];

        for i in 0..vol_size[axis] as i32 {
            // c is the 'full-convolution' index we want.
            let c = start + i;
            let mut r = Complex32::ZERO;
            // The filter f has length m=3, so we sum for j in [0..m).
            for j in 0..f_len {
                // The index in the (virtual) upsampled array we convolve with:
                let idx = c as i32 - j as i32;
                // Ensure 0 <= idx < 2*n (because upsampled has length 2*n).
                if idx >= 0 && idx < 2 * (n_coeffs as i32) {
                    // In the upsampled signal, even indices contain x[idx/2], odd indices are 0.
                    if idx % 2 == 0 {
                        let approx_idx = (idx / 2) as usize;
                        let detail_idx = (idx / 2) as usize + n_coeffs as usize;

                        let approx_idx_actual = approx_idx * decomp_stride + decomp_lane_head;
                        let detail_idx_actual = detail_idx * decomp_stride + decomp_lane_head;

                        // idx/2 is the original index in x.
                        r += decomp[approx_idx_actual] * lo_r[j as usize] +
                            decomp[detail_idx_actual] * hi_r[j as usize];
                    }
                }
            }
            let result_idx = i as usize * signal_stride + signal_lane_head;
            vol[result_idx] = r;
        }
    }
}

// returns the lane head index for a given axis and volume size enumerated over all lanes
fn lane_head(lane_idx: usize, lane_axis: usize, vol_size: &[usize; 3]) -> usize {
    assert!(lane_axis < 3);

    let n_lanes = if lane_axis == 0 {
        vol_size[1] * vol_size[2] // y-z plane
    } else if lane_axis == 1 {
        vol_size[0] * vol_size[2] // x-z plane
    } else {
        vol_size[0] * vol_size[1] // x-y plane
    };

    assert!(lane_idx < n_lanes);

    // Compute lane head index
    if lane_axis == 0 {
        // // Lane index maps to y and z coordinates: lane_idx = y + z * Ny
        // let y = lane_idx % vol_size[1];
        // let z = lane_idx / vol_size[1];
        // return (y + z * vol_size[1]) * vol_size[0];  // Start of x-lane
        return lane_idx * vol_size[0];
    } else if lane_axis == 1 {
        // Lane index maps to x and z coordinates: lane_idx = x + z * Nx
        let x = lane_idx % vol_size[0];
        let z = lane_idx / vol_size[0];
        return x + (z * vol_size[0]) * vol_size[1];  // Start of y-lane
    } else {
        // Lane index maps to x and y coordinates: lane_idx = x + y * Nx
        // let x = lane_idx % vol_size[0];
        // let y = lane_idx / vol_size[0];
        // return x + y * vol_size[0];  // Start of z-lane
        return lane_idx
    }
}


#[test]
fn test_lane_head() {
    let vol_size = [3, 3, 3];
    println!("{}", lane_head(0, 0, &vol_size));
    println!("{}", lane_head(1, 0, &vol_size));
    println!("{}", lane_head(2, 0, &vol_size));
    println!("{}", lane_head(3, 0, &vol_size));
    println!("{}", lane_head(4, 0, &vol_size));
}


// 1-D example using symmetric padding
#[test]
fn dwt_down_up() {
    let x: Vec<_> = (1..=11).map(|x| x as f32).collect();

    let w: Wavelet<f32> = Wavelet::new(WaveletType::Daubechies2);

    let lo_d = w.lo_d();
    let hi_d = w.hi_d();
    let lo_r = w.lo_r();
    let hi_r = w.hi_r();

    let f_len = 4;
    let sig_len = x.len() as i32;

    let n_coeffs = (sig_len + f_len - 1) / 2;
    let ext_len = f_len as i32 - 1;

    let mut result = vec![0.; n_coeffs as usize * 2];

    for i in 0..n_coeffs as i32 {
        let mut sa = 0.;
        let mut sd = 0.;
        for j in 0..f_len as i32 {
            // calculate virtual signal index that may extend out-of-bounds
            let virtual_idx = 2 * i - ext_len + j + 1;
            // rationalize the index by imposing the boundary condition
            //let signal_idx = symmetric_boundary_index(virtual_idx, sig_len);

            let signal_idx = if virtual_idx >= 0 && virtual_idx < sig_len {
                virtual_idx as usize
            } else if virtual_idx < 0 {
                (virtual_idx + 1).abs() as usize
            } else {
                (2 * sig_len - virtual_idx - 1) as usize
            };

            // calculate the filter index
            let filter_idx = (f_len - j - 1) as usize;
            // multiply signal with filter coefficient
            sa += lo_d[filter_idx] * x[signal_idx];
            sd += hi_d[filter_idx] * x[signal_idx];
        }
        result[i as usize] = sa;
        result[n_coeffs as usize + i as usize] = sd;
    }

    // do inverse transform to recover original data

    println!("result: {:?}", result);

    let approx = &result[0..n_coeffs as usize];
    let detail = &result[n_coeffs as usize..];

    //let n = approx.len();
    //let m = lo_r.len();
    let full_len = 2 * n_coeffs + f_len - 1;        // length of full convolution
    let keep_len = 2 * n_coeffs - f_len + 2;       // how many samples we keep (centered)
    let start = (full_len - keep_len) / 2;

    let mut recon = vec![0.; sig_len as usize];

    for i in 0..sig_len {
        // c is the 'full-convolution' index we want.
        let c = start + i;
        let mut r = 0.;
        // The filter f has length m=3, so we sum for j in [0..m).
        for j in 0..f_len {
            // The index in the (virtual) upsampled array we convolve with:
            let idx = c as i32 - j as i32;
            // Ensure 0 <= idx < 2*n (because upsampled has length 2*n).
            if idx >= 0 && idx < 2 * (n_coeffs as i32) {
                // In the upsampled signal, even indices contain x[idx/2], odd indices are 0.
                if idx % 2 == 0 {
                    // idx/2 is the original index in x.
                    r += approx[(idx / 2) as usize] * lo_r[j as usize] + detail[(idx / 2) as usize] * hi_r[j as usize];
                }
            }
        }
        recon[i as usize] = r;
    }

    println!("recon: {:?}", recon);
}


#[test]
fn conv_up() {
    let x: Vec<_> = (1..=6).map(|x| x as f32).collect();
    let f = [1., 1., 1.];

    let xlen = x.len() as i32;
    let flen = f.len() as i32;
    let sig_len = 2 * x.len() - f.len() + 2;

    let mut result = vec![];
    for i in 0..sig_len as i32 {
        let mut a = 0.;
        //let mut d = 0.;
        for j in 0..flen {
            let y = if i % 2 == 0 {
                let virtual_idx = i / 2 - flen + 1 + j;
                if virtual_idx < 0 || virtual_idx >= xlen {
                    0.
                } else {
                    x[virtual_idx as usize]
                }
            } else {
                0.
            };

            a += f[(flen - j - 1) as usize] * y;
        }
        result.push(a);
    }

    println!("result: {:?}", result);
}

// example from chat gpt
#[test]
fn direct_nested_loops() {
    // Our original signal:
    let x = [1, 2, 3, 4, 5, 6];
    // Filter:
    let f1 = [1, 1, 2, -1];
    let f2 = [0, 0, 0, 0];

    // For x of length n and filter f of length m:
    // - The (virtual) upsampled length = 2*n.
    // - The 'full' conv length would be (2*n + m - 1).
    // - The final "wkeep(..., 'c', 0)" length is L = 2*n - m + 2.
    let n = x.len();
    let m = f1.len();
    let full_len = 2 * n + m - 1;        // length of full convolution
    let keep_len = 2 * n - m + 2;       // how many samples we keep (centered)
    let start = (full_len - keep_len) / 2;
    // For n=6 and m=3, keep_len = 11, start = 1.

    // We'll store the result in a fixed-size array (no heap allocation).
    // Since we know keep_len=11 in this example, we can do:
    let mut result = vec![0; keep_len];

    // --------------------------------------------------
    // Main nested loop: compute each of the 'keep_len' outputs.
    // result[i] is the element of the "centered" sub-vector,
    // which corresponds to index (start + i) in the virtual 'full' convolution.
    // --------------------------------------------------
    for i in 0..keep_len {
        // c is the 'full-convolution' index we want.
        let c = start + i;

        let mut acc = 0;
        let mut a = 0;
        let mut d = 0;
        // The filter f has length m=3, so we sum for j in [0..m).
        for j in 0..m {
            // The index in the (virtual) upsampled array we convolve with:
            let idx = c as isize - j as isize;
            // Ensure 0 <= idx < 2*n (because upsampled has length 2*n).
            if idx >= 0 && idx < 2 * (n as isize) {
                // In the upsampled signal, even indices contain x[idx/2], odd indices are 0.
                if idx % 2 == 0 {
                    // idx/2 is the original index in x.
                    a += x[(idx / 2) as usize] * f1[j];
                    d += x[(idx / 2) as usize] * f2[j];
                }
            }
        }
        result[i] = a + d;
    }

    // Compare with the known MATLAB output:
    // wkeep(wconv1(dyadup(1:6,0), [1,1,1]), 2*6 - 3 + 2, 'c', 0)
    // => [1, 3, 2, 5, 3, 7, 4, 9, 5, 11, 6]
    let expected = [1, 3, 2, 5, 3, 7, 4, 9, 5, 11, 6];
    println!("result: {:?}", result);
    //assert_eq!(result, expected);
}

fn symmetric_boundary_index(index: i32, n: i32) -> usize {
    if index < 0 {
        (index + 1).abs() as usize
    } else if index >= n {
        (2 * n - index - 1) as usize
    } else {
        index as usize
    }
}

#[test]
fn test_symmetric_boundary() {
    let now = Instant::now();
    symmetric_boundary_index(5, 4);
    let elapsed = now.elapsed().as_nanos();
    println!("elapsed nanos: {}", elapsed);
}