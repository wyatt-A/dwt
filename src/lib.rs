mod utils;
pub mod wavelet;

use ndarray::{s, ArrayD, Axis};
use num_complex::{Complex, Complex32};
use num_traits::{Float, FromPrimitive, Signed, Zero};
use rayon::iter::{
    IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelBridge, ParallelIterator,
};
use std::{fmt::Debug, iter::Sum};
use utils::*;
use wavelet::*;

pub struct WaveDecPlanner<T> {
    signal_length: usize,
    decomp_length: usize,
    levels: Vec<usize>,
    xforms: Vec<WaveletXForm1D<T>>,
    wavelet: Wavelet<T>,
    decomp_buffer: Vec<Complex<T>>,
    signal_buffer: Vec<Complex<T>>,
}

/// single-level 3D wavelet transform
pub fn dwt3<T>(mut x: ArrayD<Complex<T>>, w: Wavelet<T>) -> ArrayD<Complex<T>>
where
    T: FromPrimitive + Copy + Signed + Sync + Send + Debug + 'static + Sum<T> + Float,
{
    let mut dims = x.shape().to_owned();

    if dims.len() != 3 {
        println!("array must have 3 dimensions for 3D wavelet transform!");
        assert!(dims.len() == 3);
    }

    for ax in 0..3 {
        let s_len = dims[ax];
        let wx = WaveletXForm1D::new(s_len, w.filt_len());

        let mut new_dims = dims.clone();
        new_dims[ax] = wx.decomp_len();

        let mut result_buff = ArrayD::<Complex<T>>::zeros(new_dims.as_slice());

        x.lanes(Axis(ax))
            .into_iter()
            .zip(result_buff.lanes_mut(Axis(ax)))
            .par_bridge()
            .for_each(|(x, mut y)| {
                let mut wx = wx.clone();
                let s = x.as_standard_layout().to_owned();
                let mut r = y.as_standard_layout().to_owned();
                wx.decompose(
                    s.as_slice().unwrap(),
                    w.lo_d(),
                    w.hi_d(),
                    r.as_slice_mut().unwrap(),
                );
                y.assign(&r);
            });
        x = result_buff;
        dims = new_dims;
    }
    x
}

/// single level inverse 3D wavelet transform. Accepts an array of subbands and returns the original
/// signal with size of specified target dims. Target dims must be equal to the original signals size
/// for an accurate reconstruction.
pub fn idwt3<T>(
    mut x: ArrayD<Complex<T>>,
    w: Wavelet<T>,
    target_dims: &[usize],
) -> ArrayD<Complex<T>>
where
    T: FromPrimitive + Copy + Signed + Sync + Send + Debug + 'static + Sum<T> + Float,
{
    let mut dims = x.shape().to_owned();

    if dims.len() != 3 {
        println!("array must have 3 dimensions for 3D wavelet transform!");
        assert!(dims.len() == 3);
    }

    let coeff_dims: Vec<_> = dims.iter().map(|d| d / 2).collect();

    for ax in 0..3 {
        let s_len = target_dims[ax];
        let wx = WaveletXForm1D::<T>::new(s_len, w.filt_len());

        let mut new_dims = dims.clone();
        new_dims[ax] = s_len;

        let mut result_buff = ArrayD::<Complex<T>>::zeros(new_dims.clone());

        x.lanes(Axis(ax))
            .into_iter()
            .zip(result_buff.lanes_mut(Axis(ax)))
            .par_bridge()
            .for_each(|(x, mut y)| {
                let mut wx = wx.clone();
                let s = x.as_standard_layout().to_owned();
                let approx = &s.as_slice().unwrap()[0..coeff_dims[ax]];
                let detail = &s.as_slice().unwrap()[coeff_dims[ax]..];
                let mut r = y.as_standard_layout().to_owned();
                wx.reconstruct(
                    approx,
                    detail,
                    w.lo_r(),
                    w.hi_r(),
                    r.as_slice_mut().unwrap(),
                );
                y.assign(&r);
            });

        x = result_buff;
        dims = new_dims;
    }

    x
}

/// single-level 3D wavelet transform. Returns a vec of subbands in order:
/// LLH, LHL, LHH, HHH, HHL, HLH, HLL, LLL
pub fn wavedec3_single_level<T>(x: ArrayD<Complex<T>>, w: Wavelet<T>) -> Vec<ArrayD<Complex<T>>>
where
    T: FromPrimitive + Copy + Signed + Sync + Send + Debug + 'static + Sum<T> + Float,
{
    let x = dwt3(x, w);

    let dims = x.shape();

    // bandsizes are half dimmensions of result
    let xb = dims[0] / 2;
    let yb = dims[1] / 2;
    let zb = dims[2] / 2;

    // slice into bands and return owned sub arrays
    let lo_lo_lo = x.slice(s![0..xb, 0..yb, 0..zb]);
    let lo_lo_hi = x.slice(s![0..xb, 0..yb, zb..]);
    let lo_hi_lo = x.slice(s![0..xb, yb.., 0..zb]);
    let lo_hi_hi = x.slice(s![0..xb, yb.., zb..]);

    let hi_hi_hi = x.slice(s![xb.., yb.., zb..]);
    let hi_hi_lo = x.slice(s![xb.., yb.., 0..zb]);
    let hi_lo_hi = x.slice(s![xb.., 0..yb, zb..]);
    let hi_lo_lo = x.slice(s![xb.., 0..yb, 0..zb]);

    vec![
        lo_lo_hi.to_owned().into_dyn(),
        lo_hi_lo.to_owned().into_dyn(),
        lo_hi_hi.to_owned().into_dyn(),
        hi_hi_hi.to_owned().into_dyn(),
        hi_hi_lo.to_owned().into_dyn(),
        hi_lo_hi.to_owned().into_dyn(),
        hi_lo_lo.to_owned().into_dyn(),
        lo_lo_lo.to_owned().into_dyn(),
    ]
}

/// single level 3D wavelet reconstruction from 8 subbands in the order:
/// LLL, HLL, HLH, HHL, HHH, LHH, LHL, LLH. The target dims must be equal to the dimensions of the
/// original signal
pub fn waverec3_single_level<T>(
    subbands: &[ArrayD<Complex<T>>],
    w: Wavelet<T>,
    target_dims: &[usize],
) -> ArrayD<Complex<T>>
where
    T: FromPrimitive + Copy + Signed + Sync + Send + Debug + 'static + Sum<T> + Float,
{
    if subbands.len() != 8 {
        println!("the number of subbands must be 8");
        assert!(subbands.len() == 8);
    }

    let dims = subbands[0].shape().to_owned();

    for i in 1..8 {
        assert_eq!(
            &dims,
            subbands[i].shape(),
            "subbands have inconsistent shapes!"
        );
    }

    // extract bandsizes for array assignment
    let xb = dims[0];
    let yb = dims[1];
    let zb = dims[2];

    let block_dims: Vec<_> = dims.iter().map(|d| d * 2).collect();

    let mut x = ArrayD::<Complex<T>>::zeros(block_dims);

    x.slice_mut(s![0..xb, 0..yb, zb..]).assign(&subbands[7]);
    x.slice_mut(s![0..xb, yb.., 0..zb]).assign(&subbands[6]);
    x.slice_mut(s![0..xb, yb.., zb..]).assign(&subbands[5]);
    x.slice_mut(s![xb.., yb.., zb..]).assign(&subbands[4]);
    x.slice_mut(s![xb.., yb.., 0..zb]).assign(&subbands[3]);
    x.slice_mut(s![xb.., 0..yb, zb..]).assign(&subbands[2]);
    x.slice_mut(s![xb.., 0..yb, 0..zb]).assign(&subbands[1]);
    x.slice_mut(s![0..xb, 0..yb, 0..zb]).assign(&subbands[0]);

    idwt3(x, w, target_dims)
}

#[derive(Clone,Debug)]
pub struct WaveDec3<T> {
    pub subbands: Vec<ArrayD<Complex<T>>>,
    pub signal_dims_per_level: Vec<Vec<usize>>,
    pub wavelet: Wavelet<T>,
}

pub fn decomp_dims<T>(shape:&[usize],w:&Wavelet<T>, num_levels:usize) -> Vec<Vec<usize>>
where
    T: FromPrimitive + Copy + Signed + Sync + Send + Debug + 'static + Sum<T> + Float,
{
    let mut decomp_dims = Vec::<Vec<usize>>::with_capacity(num_levels);

    decomp_dims.push(shape.to_vec());

    let mut new_shape = shape.to_vec();
    for _ in 1..num_levels {
        new_shape = new_shape.iter().map(|n|coeff_len(*n, w.filt_len())).collect();
        decomp_dims.push(new_shape.clone());
    }
    decomp_dims
}

pub fn subband_sizes<T>(shape:&[usize],w:&Wavelet<T>, num_levels:usize) -> Vec<Vec<usize>>
where
    T: FromPrimitive + Copy + Signed + Sync + Send + Debug + 'static + Sum<T> + Float
{

    let n_subbands = i32::from(2).pow(shape.len() as u32) as usize;

    let mut subband_sizes = Vec::<Vec<usize>>::with_capacity(num_levels*8);

    let mut new_shape = shape.to_vec();
    for _ in 0..num_levels {
        let _ = subband_sizes.pop();
        new_shape = new_shape.iter().map(|n|coeff_len(*n, w.filt_len())).collect();
        for _ in 0..n_subbands {
            subband_sizes.push(new_shape.clone());
        }
    }
    subband_sizes
}


pub fn wavedec3<T>(x: ArrayD<Complex<T>>, w: Wavelet<T>, num_levels: usize) -> WaveDec3<T>
where
    T: FromPrimitive + Copy + Signed + Sync + Send + Debug + 'static + Sum<T> + Float,
{
    assert!(num_levels != 0, "num_levels must be greater than 0!");

    let mut signal_dims = vec![];
    signal_dims.push(x.shape().to_owned());

    // returns vector of subbands (7 per level, except for the last level)
    let mut x = wavedec3_single_level(x, w.clone());

    // for each level, pop the LLL subband, decompose it, then push those subbands to the stack
    for _ in 1..num_levels {
        let input = x.pop().unwrap().into_dyn();
        signal_dims.push(input.shape().to_owned());
        let mut y = wavedec3_single_level(input, w.clone());
        x.append(&mut y);
    }

    WaveDec3 {
        subbands: x,
        signal_dims_per_level: signal_dims,
        wavelet: w,
    }
}

/// Multi-level 3D wavelet reconstruction from a previous deconstruction
pub fn waverec3<T>(mut dec: WaveDec3<T>) -> ArrayD<Complex<T>>
where
    T: FromPrimitive + Copy + Signed + Sync + Send + Debug + 'static + Sum<T> + Float,
{
    dec.signal_dims_per_level.reverse();
    for s in dec.signal_dims_per_level {
        let subbands: Vec<_> = (0..8).map(|_| dec.subbands.pop().unwrap()).collect();
        let rec = waverec3_single_level(&subbands, dec.wavelet.clone(), &s);
        dec.subbands.push(rec)
    }

    let rec = dec.subbands.pop().unwrap();
    assert!(
        dec.subbands.len() == 0,
        "not all subbands were reconstructed!"
    );
    rec
}

impl<T> WaveDecPlanner<T>
where
    T: FromPrimitive + Copy + Signed + Sync + Send + Debug + 'static + Sum<T> + Float,
{
    pub fn new(signal_length: usize, n_levels: usize, wavelet: Wavelet<T>) -> Self {
        if n_levels < 1 {
            panic!("number of levels must be 1 or more");
        }
        let mut levels = vec![0; n_levels + 2];
        levels[n_levels + 1] = signal_length;
        let mut xforms = vec![];
        let mut sig_len = signal_length;
        for level in 0..n_levels {
            let w = WaveletXForm1D::<T>::new(sig_len, wavelet.lo_d().len());
            levels[n_levels - level] = w.coeff_len;
            sig_len = w.coeff_len;
            xforms.push(w);
        }
        *levels.first_mut().unwrap() = xforms.last().unwrap().coeff_len;

        let mut decomp_len = xforms.iter().fold(0, |acc, x| acc + x.coeff_len);
        decomp_len += xforms.last().unwrap().coeff_len;

        let decomp = vec![Complex::<T>::zero(); decomp_len];
        let signal = decomp.clone();

        Self {
            signal_length,
            decomp_length: decomp_len,
            levels: levels,
            xforms,
            wavelet,
            decomp_buffer: decomp,
            signal_buffer: signal,
        }
    }

    pub fn decomp_len(&self) -> usize {
        self.decomp_length
    }

    pub fn process(&mut self, signal: &[Complex<T>], result: &mut [Complex<T>]) {
        let signal_energy = signal.par_iter().map(|x| x.norm_sqr()).sum::<T>();

        let mut stop = self.decomp_buffer.len();

        let lo_d = &self.wavelet.lo_d();
        let hi_d = self.wavelet.hi_d();

        self.decomp_buffer[0..self.signal_length].copy_from_slice(&signal[0..self.signal_length]);
        let mut rl = 0;
        let mut ru = self.signal_length;
        for xform in self.xforms.iter_mut() {
            let start = stop - xform.decomp_len;
            self.signal_buffer[rl..ru].copy_from_slice(&self.decomp_buffer[rl..ru]);
            xform.decompose(
                &self.signal_buffer[rl..ru],
                lo_d,
                hi_d,
                &mut self.decomp_buffer[start..stop],
            );
            rl = start;
            ru = start + xform.coeff_len;
            stop -= xform.coeff_len;
        }

        //println!("decomp = {:#?}",self.decomp_buffer);

        result.copy_from_slice(&self.decomp_buffer);

        let result_energy = result.par_iter().map(|x| x.norm_sqr()).sum::<T>();

        if !result_energy.is_zero() {
            let scale = (signal_energy / result_energy).sqrt();
            result.par_iter_mut().for_each(|x| *x = *x * scale);
        }

        //self.decomp_buffer.clone()
    }
}

pub struct WaveRecPlanner<T> {
    levels: Vec<usize>,
    xforms: Vec<(usize, usize, WaveletXForm1D<T>)>,
    signal_length: usize,
    signal_buffer: Vec<Complex<T>>,
    approx_buffer: Vec<Complex<T>>,
    wavelet: Wavelet<T>,
}

impl<T> WaveRecPlanner<T>
where
    T: FromPrimitive + Copy + Signed + Sync + Send + Debug + 'static + Sum<T> + Float,
{
    pub fn new(dec_planner: &WaveDecPlanner<T>) -> Self {
        let levels = dec_planner.levels.to_owned();
        let signal_length = dec_planner.signal_length;
        let signal = vec![Complex::<T>::zero(); signal_length];
        let approx = vec![Complex::<T>::zero(); signal_length];
        let filt_len = dec_planner.wavelet.filt_len();
        let xforms: Vec<_> = levels[1..]
            .windows(2)
            .map(|x| {
                (
                    x[0],                                     // number of approx coeffs
                    x[1],                                     // signal length to reconstruct
                    WaveletXForm1D::<T>::new(x[1], filt_len), // wavelet transform handler
                )
            })
            .collect();
        Self {
            levels: dec_planner.levels.to_owned(),
            xforms,
            signal_length,
            signal_buffer: signal,
            approx_buffer: approx,
            wavelet: dec_planner.wavelet.clone(),
        }
    }

    pub fn process(&mut self, decomposed: &[Complex<T>], result: &mut [Complex<T>]) {
        let decomp_energy = decomposed.par_iter().map(|x| x.norm_sqr()).sum::<T>();

        self.approx_buffer[0..self.levels[0]].copy_from_slice(&decomposed[0..self.levels[0]]);
        let lo_r = self.wavelet.lo_r();
        let hi_r = self.wavelet.hi_r();
        let mut start = self.levels[1];
        for (n_coeffs, sig_len, w) in self.xforms.iter_mut() {
            let detail = &decomposed[start..(start + *n_coeffs)];
            start += *n_coeffs;
            w.reconstruct(
                &self.approx_buffer[0..*n_coeffs],
                detail,
                lo_r,
                hi_r,
                &mut self.signal_buffer[0..*sig_len],
            );
            self.approx_buffer[0..*sig_len].copy_from_slice(&self.signal_buffer[0..*sig_len]);
        }
        //self.signal_buffer.clone()
        result.copy_from_slice(&self.signal_buffer);

        let result_energy = result.par_iter().map(|x| x.norm_sqr()).sum::<T>();

        if !result_energy.is_zero() {
            let scale = (decomp_energy / result_energy).sqrt();
            result.par_iter_mut().for_each(|x| *x = *x * scale);
        }
    }
}

#[derive(Clone)]
pub struct WaveletXForm1D<T> {
    /// length of filter coefficients
    filt_len: usize,
    /// length of signal
    sig_len: usize,
    /// length of decomposed signal
    decomp_len: usize,
    /// length of symmetric signal extension
    decomp_ext_len: usize,
    /// length of extended signal
    decomp_ext_result_len: usize,
    /// length of signal/filter convolution result
    decomp_conv_len: usize,
    /// length of detail/approx coefficents
    coeff_len: usize,
    /// length of reconstruction convolutions
    recon_conv_len: usize,
    /// lower index of convolution result center
    recon_conv_center_lidx: usize,
    /// upper index of convolution result center
    recon_conv_center_uidx: usize,

    decomp_conv_valid_lidx: usize,
    decomp_conv_valid_uidx: usize,

    decomp_scratch1: Vec<Complex<T>>,
    decomp_scratch2: Vec<Complex<T>>,
    decomp_scratch3: Vec<Complex<T>>,
    decomp_scratch4: Vec<Complex<T>>,

    recon_scratch1: Vec<Complex<T>>,
    recon_scratch2: Vec<Complex<T>>,
    recon_scratch3: Vec<Complex<T>>,
    recon_scratch4: Vec<Complex<T>>,

    recon_upsample_scratch: Vec<Complex<T>>,
}

pub fn coeff_len(sig_len:usize,filt_len:usize) -> usize {
    (filt_len + sig_len - 1) / 2
}

impl<T> WaveletXForm1D<T>
where
    T: FromPrimitive + Copy + Signed + Sync + Send + Debug + 'static,
{
    pub fn new(sig_len: usize, filt_len: usize) -> Self {
        let decomp_ext_len = filt_len - 1;
        let decomp_len = 2 * ((filt_len + sig_len - 1) / 2);
        let decomp_ext_result_len = 2 * decomp_ext_len + sig_len;
        let decomp_conv_len = conv_len(decomp_ext_result_len, filt_len);
        let coeff_len = (filt_len + sig_len - 1) / 2;

        let upsamp_len = coeff_len * 2 - 1;
        let recon_conv_len = conv_len(upsamp_len, filt_len);
        let conv_center = conv_center(recon_conv_len, sig_len);
        let decomp_conv_valid = conv_valid_idx(sig_len, filt_len);

        Self {
            filt_len,
            sig_len,
            decomp_len,
            decomp_ext_len,
            decomp_ext_result_len,
            decomp_conv_len,
            coeff_len,
            recon_conv_len,
            recon_conv_center_lidx: conv_center.0,
            recon_conv_center_uidx: conv_center.1,
            decomp_conv_valid_lidx: decomp_conv_valid.0,
            decomp_conv_valid_uidx: decomp_conv_valid.1,
            decomp_scratch1: vec![Complex::<T>::zero(); decomp_conv_len],
            decomp_scratch2: vec![Complex::<T>::zero(); decomp_conv_len],
            decomp_scratch3: vec![Complex::<T>::zero(); decomp_conv_len],
            decomp_scratch4: vec![Complex::<T>::zero(); decomp_conv_len],
            recon_scratch1: vec![Complex::<T>::zero(); recon_conv_len],
            recon_scratch2: vec![Complex::<T>::zero(); recon_conv_len],
            recon_scratch3: vec![Complex::<T>::zero(); recon_conv_len],
            recon_scratch4: vec![Complex::<T>::zero(); recon_conv_len],
            recon_upsample_scratch: vec![Complex::<T>::zero(); upsamp_len],
        }
    }

    pub fn coeff_len(&self) -> usize {
        self.coeff_len
    }

    pub fn decomp_len(&self) -> usize {
        self.decomp_len
    }

    pub fn decomp_buffer(&self) -> Vec<Complex<T>> {
        vec![Complex::<T>::zero(); self.decomp_len]
    }

    pub fn recon_buffer(&self, sig_len: usize) -> Vec<Complex<T>> {
        vec![Complex::<T>::zero(); sig_len]
    }

    pub fn decompose(
        &mut self,
        signal: &[Complex<T>],
        lo_d: &[T],
        hi_d: &[T],
        decomp: &mut [Complex<T>],
    ) {
        symm_ext(signal, self.decomp_ext_len, &mut self.decomp_scratch1);
        //conv1d(&self.decomp_scratch1[0..self.decomp_ext_result_len], lo_d,&mut self.decomp_scratch3,&mut self.decomp_scratch4, &mut self.decomp_scratch2);
        conv_direct(
            &self.decomp_scratch1[0..self.decomp_ext_result_len],
            lo_d,
            &mut self.decomp_scratch2,
        );
        downsample2(
            &self.decomp_scratch2[conv_valid_range(self.decomp_ext_result_len, self.filt_len)],
            &mut decomp[0..self.coeff_len],
        );
        //downsample2(&self.decomp_scratch2[self.decomp_conv_valid_lidx..self.decomp_conv_valid_uidx], &mut decomp[0..self.coeff_len]);
        //conv1d(&self.decomp_scratch1[0..self.decomp_ext_result_len], hi_d,&mut self.decomp_scratch3,&mut self.decomp_scratch4, &mut self.decomp_scratch2);
        conv_direct(
            &self.decomp_scratch1[0..self.decomp_ext_result_len],
            hi_d,
            &mut self.decomp_scratch2,
        );
        downsample2(
            &self.decomp_scratch2[conv_valid_range(self.decomp_ext_result_len, self.filt_len)],
            &mut decomp[self.coeff_len..],
        );
    }

    pub fn reconstruct(
        &mut self,
        approx: &[Complex<T>],
        detail: &[Complex<T>],
        lo_r: &[T],
        hi_r: &[T],
        signal: &mut [Complex<T>],
    ) {
        let conv_center = conv_center(self.recon_conv_len, signal.len());
        upsample_odd(approx, &mut self.recon_upsample_scratch);
        //conv1d(&self.recon_upsample_scratch, lo_r, &mut self.recon_scratch3, &mut self.recon_scratch4, &mut self.recon_scratch1);
        conv_direct(&self.recon_upsample_scratch, lo_r, &mut self.recon_scratch1);
        upsample_odd(detail, &mut self.recon_upsample_scratch);
        //conv1d(&self.recon_upsample_scratch, &hi_r, &mut self.recon_scratch3, &mut self.recon_scratch4, &mut self.recon_scratch2);
        conv_direct(
            &self.recon_upsample_scratch,
            &hi_r,
            &mut self.recon_scratch2,
        );
        let a = &self.recon_scratch1[conv_center.0..conv_center.1];
        let d = &self.recon_scratch2[conv_center.0..conv_center.1];
        signal.iter_mut().enumerate().for_each(|(idx, x)| {
            *x = a[idx] + d[idx];
        });
    }
}

/// Returns the maximum number of wavelet decomposition levels to avoid boundary effects
pub fn w_max_level(sig_len: usize, filt_len: usize) -> usize {
    if filt_len <= 1 {
        return 0;
    }
    (sig_len as f32 / (filt_len as f32 - 1.)).log2() as usize
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use num_complex::{Complex64, ComplexFloat};
    use num_traits::One;
    use rayon::{iter::IndexedParallelIterator, slice::ParallelSliceMut};

    use rand::{self, Rng};
    use super::*;

    fn rand_array(n:usize) -> Vec<Complex32> {
        let mut rng = rand::thread_rng();
        let mut data = Vec::<Complex32>::with_capacity(n);
        for _ in 0..n {
            data.push(Complex32::new(
                rng.gen_range((-1.)..1.),
                rng.gen_range((-1.)..1.),
            ))
        }
        data
    }

    #[test]
    fn single_level_1d() {
        let x = rand_array(200);
        
        let w1 = Wavelet::new(WaveletType::Daubechies2);
        let w2 = Wavelet::new(WaveletType::Daubechies3);

        let mut xform1 = WaveletXForm1D::new(x.len(),w1.filt_len());
        let mut xform2 = WaveletXForm1D::new(x.len(),w2.filt_len());

        let mut d1 = xform1.decomp_buffer();
        let mut d2 = xform2.decomp_buffer();

        xform1.decompose(&x, w1.lo_d(), w1.hi_d(), &mut d1);
        xform2.decompose(&x, w2.lo_d(), w2.hi_d(), &mut d2);

        let mut r1 = xform1.recon_buffer(x.len());
        let mut r2 = xform2.recon_buffer(x.len());

        xform1.reconstruct(&d1[0 .. d1.len()/2],&d1[d1.len()/2 ..],w1.lo_r(), w1.hi_r(), &mut r1);
        xform2.reconstruct(&d2[0 .. d2.len()/2],&d2[d2.len()/2 ..],w2.lo_r(), w2.hi_r(), &mut r2);

        let max_err1 = r1.iter().zip(x.iter()).map(|(x,y)| (*x - *y).abs()).max_by(|x,y|x.partial_cmp(y).unwrap()).unwrap();

        let max_err2 = r2.iter().zip(x.iter()).map(|(x,y)| (*x - *y).abs()).max_by(|x,y|x.partial_cmp(y).unwrap()).unwrap();

        assert!(max_err1 < 1E-6);
        assert!(max_err2 < 1E-6);
    }

    #[test]
    fn decomp_3d() {
        //cargo test --release --package dwt --lib -- tests::decomp_3d --exact --nocapture
        let dims = [78,48,46];
        let min_dim = *dims.iter().min().unwrap();
        let n = dims.iter().product();

        let w = Wavelet::new(WaveletType::Daubechies3);

        let r = rand_array(n);

        let x = ArrayD::from_shape_vec(dims.as_slice(), r).unwrap();

        let n_lev = w_max_level(min_dim, w.filt_len());

        println!("running decomp and recon ...");
        let now = Instant::now();
        let dec = wavedec3(x.clone(), w, n_lev);
        let rec = waverec3(dec);
        let dur = now.elapsed().as_millis();
        println!("decomp and recon took {} ms",dur);

        let err = (x - rec).map(|x|x.abs());
        
        let max_err = *err.iter().max_by(|x,y|x.partial_cmp(y).unwrap()).unwrap();

        println!("max error: {}",max_err);
        assert!(max_err < 1E-6);

    }

    #[test]
    fn decomp_size() {
        let dims = [78,48,46];
        let n = dims.iter().product();
        let w = Wavelet::new(WaveletType::Daubechies3);
        let r = rand_array(n);
        let x = ArrayD::from_shape_vec(dims.as_slice(), r).unwrap();

        let dd = decomp_dims(x.shape(), &w, 4);

        let dec = wavedec3(x.clone(), w.clone(), 4);


        for x in &dec.subbands {
            println!("{:?}",x.shape());
        }

        for x in subband_sizes(x.shape(), &w, 4) {
            println!("{:?}",x);
        }

        assert_eq!(dec.signal_dims_per_level,dd);

    }

    
}
