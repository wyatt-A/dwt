use num_complex::Complex;
use num_traits::{FromPrimitive, Signed, Zero};
use rustfft::FftPlannerAvx;
use std::{fmt::Debug, ops::Range};

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

/// 1-D convolution based on the fast fourier transform. Buffers for the fft are explicitly passed and must be the same size
/// as the result buffer. The size of the result buffer needs to be :
///  signal.len() + filter.len() - 1
// pub fn conv1d<T>(
//     signal: &[T],
//     filter: &[T],
//     s_buff1: &mut [T],
//     s_buff2: &mut [T],
//     result: &mut [T],
// ) where
//     T: FromPrimitive + Copy + Signed + Sync + Send + Debug + 'static,
// {
//     let n = result.len();

//     s_buff1.iter_mut().for_each(|x| *x = T::zero());
//     s_buff2.iter_mut().for_each(|x| *x = T::zero());

//     s_buff1.iter_mut().zip(signal.iter()).for_each(|(x, s)| {
//         *x = *s;
//     });

//     s_buff2.iter_mut().zip(filter.iter()).for_each(|(x, f)| {
//         *x = *f
//     });

//     let mut fp = FftPlannerAvx::new().unwrap();
//     let fft = fp.plan_fft_forward(n);
//     let ifft = fp.plan_fft_inverse(n);
//     fft.process(s_buff1);
//     fft.process(s_buff2);

//     s_buff1
//         .into_iter()
//         .zip(s_buff2.into_iter())
//         .zip(result.iter_mut())
//         .for_each(|((x, y), r)| {
//             *r = *x * *y;
//         });

//     ifft.process(result);

//     let scale = T::one() / T::from_usize(n).unwrap();

//     result.iter_mut().for_each(|x| {
//         *x = *x * scale;
//     })
// }

/// Retuns the length of the resulting convolution given the signal length and the filter length
pub fn conv_len(sig_len: usize, filt_len: usize) -> usize {
    sig_len + filt_len - 1
}
