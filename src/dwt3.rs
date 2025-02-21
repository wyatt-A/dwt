use crate::wavelet::{Wavelet, WaveletFilter, WaveletType};
use cfl::ndarray_stats::QuantileExt;
use ndarray::{Array3, ShapeBuilder};
use num_complex::{Complex32, ComplexFloat};
use std::time::Instant;

#[test]
fn test_dwt3_axis() {
    for axis in 0..3 {
        let vol_size = [512, 284, 228];
        let w: Wavelet<f32> = Wavelet::new(WaveletType::Daubechies2);

        let mut x = Array3::<Complex32>::from_shape_fn(vol_size.f(), |(i, j, k)| Complex32::new((i + j + k) as f32, 0.0));
        let x_original = x.clone();

        let n_coeffs_x = sub_band_size(vol_size[axis], w.filt_len());
        let result_size_axis = 2 * n_coeffs_x;

        let mut decomp_size = vol_size.clone();
        decomp_size[axis] = result_size_axis;

        let mut r = Array3::<Complex32>::zeros(decomp_size.f());

        let shift = [0, 0, 0];

        let now = Instant::now();
        dwt3_axis(
            x.as_slice_memory_order().unwrap(),
            r.as_slice_memory_order_mut().unwrap(),
            &vol_size,
            &decomp_size,
            axis,
            w.lo_d(),
            w.hi_d(),
            &shift,
        );

        //cfl::dump_magnitude("x", &r.into_dyn());

        idwt3_axis(
            x.as_slice_memory_order_mut().unwrap(),
            r.as_slice_memory_order().unwrap(),
            &vol_size,
            &decomp_size,
            axis,
            w.lo_r(),
            w.hi_r(),
            &shift,
        );

        let elapsed = now.elapsed();
        println!("xform took {} ms", elapsed.as_millis());

        let diff = &x - &x_original;
        let max_error = *diff.map(|x| x.abs()).max().unwrap();

        x.mapv_inplace(|x| Complex32::new(x.re.round(), x.im.round()));

        println!("max error {:?}", max_error);
        assert_eq!(x, x_original);

        //cfl::dump_magnitude("x", &x.into_dyn());
    }
}

#[test]
fn test_dwt3_all() {
    let input_size = [512, 256, 128];
    let mut x = Array3::<Complex32>::from_shape_fn(input_size.f(), |(i, j, k)| Complex32::new((i + j + k) as f32, 0.0));
    let x_original = x.clone();

    let now = Instant::now();
    let w: Wavelet<f32> = Wavelet::new(WaveletType::Daubechies2);
    let result_size = result_size(&input_size, w.filt_len());
    let tmp1_size = [result_size[0], input_size[1], input_size[2]];
    let tmp2_size = [result_size[0], result_size[1], input_size[2]];

    let mut tmp1 = Array3::zeros(tmp1_size.f());
    let mut tmp2 = Array3::zeros(tmp2_size.f());
    let mut result = Array3::zeros(result_size.f());

    let shift = [10, -3, 6];

    dwt3_axis(
        x.as_slice_memory_order().unwrap(),
        tmp1.as_slice_memory_order_mut().unwrap(),
        &input_size,
        &tmp1_size,
        0,
        w.lo_d(),
        w.hi_d(),
        &shift,
    );

    dwt3_axis(
        tmp1.as_slice_memory_order().unwrap(),
        tmp2.as_slice_memory_order_mut().unwrap(),
        &tmp1_size,
        &tmp2_size,
        1,
        w.lo_d(),
        w.hi_d(),
        &shift,
    );

    dwt3_axis(
        tmp2.as_slice_memory_order().unwrap(),
        result.as_slice_memory_order_mut().unwrap(),
        &tmp2_size,
        &result_size,
        2,
        w.lo_d(),
        w.hi_d(),
        &shift,
    );


    idwt3_axis(
        tmp2.as_slice_memory_order_mut().unwrap(),
        result.as_slice_memory_order().unwrap(),
        &tmp2_size,
        &result_size,
        2,
        w.lo_r(),
        w.hi_r(),
        &shift,
    );

    idwt3_axis(
        tmp1.as_slice_memory_order_mut().unwrap(),
        tmp2.as_slice_memory_order().unwrap(),
        &tmp1_size,
        &tmp2_size,
        1,
        w.lo_r(),
        w.hi_r(),
        &shift,
    );

    idwt3_axis(
        x.as_slice_memory_order_mut().unwrap(),
        tmp1.as_slice_memory_order().unwrap(),
        &input_size,
        &tmp1_size,
        0,
        w.lo_r(),
        w.hi_r(),
        &shift,
    );

    let elapsed = now.elapsed();

    println!("3D wavelet decomp took {} ms", elapsed.as_millis());

    x.mapv_inplace(|x| Complex32::new(x.re.round(), x.im.round()));

    assert_eq!(x, x_original);

    cfl::dump_magnitude("x", &x.into_dyn());
}

/// returns the full result size of the 3D wavelet decomposition given the input size and the
/// filter length
pub fn result_size(vol_size: &[usize; 3], f_len: usize) -> [usize; 3] {
    let mut r = [0, 0, 0];
    vol_size.iter().zip(r.iter_mut()).for_each(|(&d, r)| {
        let tmp = (d + f_len - 1) / 2;
        *r = tmp * 2;
    });
    r
}

/// returns the size of the sub band given the signal length and filter length. This is the size
/// of the approximation and detail coefficients.
pub fn sub_band_size(signal_length: usize, filter_length: usize) -> usize {
    (signal_length + filter_length - 1) / 2
}


//fn lane_stride(vol_size: &[usize; 3], axis: usize) -> usize {}

fn dwt3_axis(vol_data: &[Complex32], decomp: &mut [Complex32], vol_size: &[usize; 3], decomp_size: &[usize; 3], axis: usize, lo_d: &[f32], hi_d: &[f32], rand_shift: &[isize; 3]) {
    assert!(axis < 3, "axis out of bounds");

    assert_eq!(
        lo_d.len(),
        hi_d.len(),
        "approximation and detail filter coefficients have different lengths"
    );

    let f_len = lo_d.len();

    // calculate the expected size of the decomposition and check consistency with supplied decomp size
    let mut expected_size = vol_size.to_vec();
    expected_size[axis] = result_size(vol_size, f_len)[axis];
    assert_eq!(
        decomp_size,
        expected_size.as_slice(),
        "mismatch between expected and supplied decomposition size"
    );

    // assert that the expected size and the number of decomp coefficients is consistent
    assert_eq!(
        expected_size.into_iter().product::<usize>(),
        decomp.len(),
        "mismatch between expected and supplied decomposition buffer size"
    );

    // assert that the volume size is consistent with the number of elements in vol data
    assert_eq!(
        vol_size.iter().product::<usize>(),
        vol_data.len(),
        "mismatch between expected and supplied volume buffer size"
    );

    let n_coeffs = sub_band_size(vol_size[axis], f_len) as i32;

    let f_len = f_len as i32;
    // signal padding size for both sides of lane
    let signal_extension_len = f_len - 1;
    let sig_len = vol_size[axis] as i32;

    // this is the jump to get from one signal element to the next across an axis
    let signal_stride = lane_stride(vol_size, axis);
    let decomp_stride = lane_stride(decomp_size, axis);

    let n_lanes = num_lanes(vol_size, axis);

    for lane in 0..n_lanes {
        // calculate the volume-relative lane head addresses for both source and destination.
        // This is the starting point for the lane
        let signal_lane_head = lane_head(lane, axis, vol_size);
        let result_lane_head = lane_head(lane, axis, decomp_size);

        // output loop for calculating coefficients
        for i in 0..n_coeffs {

            // initialize the approximation and detail coefficients to calculate
            let mut a = Complex32::ZERO;
            let mut d = Complex32::ZERO;

            // filter coefficient loop for performing dot product
            for j in 0..f_len {

                // calculate virtual signal index to multiply with filter coefficient
                let virtual_idx = 2 * i - signal_extension_len + j + 1;

                // rectify the virtual index to a valid index by imposing symmetric boundary reflection.
                // this is local to the lane
                let sample_address_lane = symmetric_boundary_index(virtual_idx, sig_len);
                // let sample_address_lane = if virtual_idx >= 0 && virtual_idx < sig_len {
                //     virtual_idx as usize
                // } else if virtual_idx < 0 {
                //     (virtual_idx + 1).abs() as usize
                // } else {
                //     (2 * sig_len - virtual_idx - 1) as usize
                // };

                // calculate the filter index
                let filter_idx = (f_len - j - 1) as usize;

                // calculate the linear sample address to read from volume data
                let sample_address_vol = sample_address_lane * signal_stride + signal_lane_head;

                // re-calculate the sample address to account for a 3-D circular shift
                let sample_address_vol = shift_address(sample_address_vol, vol_size, rand_shift);

                // multiply signal with filter coefficient

                a += lo_d[filter_idx] * vol_data[sample_address_vol];
                d += hi_d[filter_idx] * vol_data[sample_address_vol];
            }

            let result_index_a = i as usize * decomp_stride + result_lane_head;
            let result_index_d = (i + n_coeffs) as usize * decomp_stride + result_lane_head;

            decomp[result_index_a] = a;
            decomp[result_index_d] = d;
        }
    }
}

fn idwt3_axis(vol: &mut [Complex32], decomp: &[Complex32], vol_size: &[usize; 3], decomp_size: &[usize; 3], axis: usize, lo_r: &[f32], hi_r: &[f32], rand_shift: &[isize; 3]) {
    let decomp_stride = lane_stride(decomp_size, axis);
    let signal_stride = lane_stride(vol_size, axis);

    let n_lanes = num_lanes(decomp_size, axis);

    let n_coeffs = decomp_size[axis] as i32 / 2;

    assert_eq!(lo_r.len(), hi_r.len());

    let f_len = lo_r.len() as i32;

    let full_len = 2 * n_coeffs + f_len - 1;        // length of full convolution
    let keep_len = 2 * n_coeffs - f_len + 2;       // how many samples we keep (centered)
    let start = (full_len - keep_len) / 2;

    for lane in 0..n_lanes {
        let decomp_lane_head = lane_head(lane, axis, decomp_size);
        let signal_lane_head = lane_head(lane, axis, vol_size);

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
            let sample_address_vol = i as usize * signal_stride + signal_lane_head;

            // re-calculate the sample address to account for the circular shift
            let sample_address_vol = shift_address(sample_address_vol, vol_size, rand_shift);

            vol[sample_address_vol] = r;
        }
    }
}

/// returns the number of lanes along an axis
fn num_lanes(vol_size: &[usize; 3], axis: usize) -> usize {
    assert!(axis < 3);
    if axis == 0 {
        vol_size[1] * vol_size[2]
    } else if axis == 1 {
        vol_size[0] * vol_size[2]
    } else {
        vol_size[0] * vol_size[1]
    }
}

/// returns the stride to jump to one element of a lane to the next
fn lane_stride(vol_size: &[usize; 3], axis: usize) -> usize {
    assert!(axis < 3);
    if axis == 0 {
        1
    } else if axis == 1 {
        vol_size[0]
    } else {
        vol_size[0] * vol_size[1]
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

#[inline(always)]
fn symmetric_boundary_index(index: i32, n: i32) -> usize {
    if index < 0 {
        (index + 1).abs() as usize
    } else if index >= n {
        (2 * n - index - 1) as usize
    } else {
        index as usize
    }
}

// let sample_address_lane = if virtual_idx >= 0 && virtual_idx < sig_len {
// virtual_idx as usize
// } else if virtual_idx < 0 {
// (virtual_idx + 1).abs() as usize
// } else {
// (2 * sig_len - virtual_idx - 1) as usize
// };

/// returns a new column-major address after some 3-D shift
// #[inline]
// fn shift_address(address: usize, vol_size: &[usize; 3], shift: &[isize; 3]) -> usize {
//     let mut coord = index_to_subscript_col_maj3(address, vol_size);
//     coord.iter_mut().zip(shift).zip(vol_size).for_each(|((c, &o), &l)| {
//         *c = (*c as isize + o).checked_rem_euclid(l as isize).unwrap() as usize;
//     });
//     subscript_to_index_col_maj3(&coord, vol_size)
// }

#[inline] // Suggest inlining for performance
fn shift_address(address: usize, vol_size: &[usize; 3], shift: &[isize; 3]) -> usize {
    let nz = vol_size[0];
    let ny = vol_size[1];
    let nx = vol_size[2];

    let z = address % nz;
    let remainder = address / nz;
    let y = remainder % ny;
    let x = remainder / ny;

    let new_z = ((z as isize + shift[0]).rem_euclid(nz as isize)) as usize;
    let new_y = ((y as isize + shift[1]).rem_euclid(ny as isize)) as usize;
    let new_x = ((x as isize + shift[2]).rem_euclid(nx as isize)) as usize;

    new_z + nz * (new_y + ny * new_x)
}

#[test]
fn test_symmetric_boundary() {
    let now = Instant::now();
    symmetric_boundary_index(5, 4);
    let elapsed = now.elapsed().as_nanos();
    println!("elapsed nanos: {}", elapsed);
}