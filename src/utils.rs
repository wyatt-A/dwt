use num_complex::Complex;
use num_traits::{FromPrimitive, Signed, Zero};
use std::ops::Range;
use std::fmt::Debug;

/// Performs a direct convolution of the input with the kernel. The length of the result is:
/// input.len() + kernel.len() - 1
pub fn conv_direct<T>(input: &[Complex<T>], kernel: &[T], result: &mut [Complex<T>])
where
    T: FromPrimitive + Copy + Signed + Sync + Send + Debug + 'static,
{
    result.iter_mut().for_each(|x| *x = Complex::<T>::zero());

    let input_len = input.len();
    let kernel_len = kernel.len();
    let result_len = input_len + kernel_len - 1;

    for i in 0..result_len {
        for j in 0..kernel_len {
            if i >= j && i - j < input_len {
                result[i] = result[i] + input[i - j] * kernel[j];
            }
        }
    }
}

/// Downsamples the signal by removing every odd index
pub fn downsample2<T>(sig: &[Complex<T>], downsampled: &mut [Complex<T>])
where
    T: FromPrimitive + Copy + Signed + Sync + Send + Debug + 'static,
{
    downsampled
        .iter_mut()
        .enumerate()
        .for_each(|(idx, r)| *r = sig[2 * idx + 1])
}

/// Upsamples the signal by inserting 0s every odd index
pub fn upsample_odd<T>(sig: &[Complex<T>], upsampled: &mut [Complex<T>])
where
    T: FromPrimitive + Copy + Signed + Sync + Send + Debug + 'static,
{

    //println!("sig length: {}",sig.len());
    //println!("upsample length: {}",upsampled.len());

    upsampled.iter_mut().for_each(|x| *x = Complex::<T>::zero());
    sig.iter().enumerate().for_each(|(idx, x)| {
        upsampled[2 * idx] = *x;
    })
}

/// Pads the signal array with symmetric boundaries of length a. The result array is:
/// sig.len() + 2 * a
pub fn symm_ext<T>(sig: &[Complex<T>], a: usize, oup: &mut [Complex<T>])
where
    T: FromPrimitive + Copy + Signed + Sync + Send + Debug + 'static,
{
    let len = sig.len();

    // Copy the original signal to the middle of the extended array
    for i in 0..len {
        oup[a + i] = sig[i];
    }

    let mut len2 = len;

    // Symmetrically extend on both sides
    for i in 0..a {
        let temp1 = oup[a + i];
        let temp2 = oup[a + len2 - 1 - i];
        oup[a - 1 - i] = temp1;
        oup[len2 + a + i] = temp2;
    }
}

/// Pads the signal array with periodic boundaries of length a. The result array is:
/// sig.len() + 2 * a
pub fn per_ext<T>(sig: &[Complex<T>], a: usize, oup: &mut [Complex<T>])
where
    T: FromPrimitive + Copy + Signed + Sync + Send + Debug + 'static,
{
    let len = sig.len();
    let mut len2 = len;
    let mut temp1 = Complex::<T>::zero();
    let mut temp2 = Complex::<T>::zero();

    for i in 0..len {
        oup[a + i] = sig[i];
    }

    if len % 2 != 0 {
        len2 = len + 1;
        oup[a + len] = sig[len - 1];
    }

    for i in 0..a {
        temp1 = oup[a + i];
        temp2 = oup[a + len2 - 1 - i];
        oup[a - 1 - i] = temp2;
        oup[len2 + a + i] = temp1;
    }
}

/// return the index range of the valid portion of the convolution
pub fn conv_valid_range(sig_len: usize, filt_len: usize) -> Range<usize> {
    if filt_len < 1 {
        panic!("filter length must be greater than 0");
    }
    (filt_len - 1)..sig_len
}

/// return the lower and upper (non-inclusive) index of the valid portion of the convolution
pub fn conv_valid_idx(sig_len: usize, filt_len: usize) -> (usize, usize) {
    if filt_len < 1 {
        panic!("filter length must be greater than 0");
    }
    (filt_len - 1, sig_len)
}

/// Returns the lower and upper (non-inclusive) index of the central portion of the convolution
pub fn conv_center(sig_len: usize, center_len: usize) -> (usize, usize) {
    let f = (sig_len - center_len) / 2;
    (f, f + center_len)
}

/// Retuns the length of the resulting convolution given the signal length and the filter length
pub fn conv_len(sig_len: usize, filt_len: usize) -> usize {
    sig_len + filt_len - 1
}
