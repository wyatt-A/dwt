use crate::{
    utils::{
        conv_center, conv_direct, conv_len, conv_valid_idx, conv_valid_range, downsample2,
        symm_ext, upsample_odd,
    },
    wavelet::{Wavelet, WaveletFilter, WaveletType},
};
use ndarray::{s, Array3, ArrayD, Axis};
use num_complex::{Complex, Complex32};
use num_traits::{Float, FromPrimitive, Signed, Zero};
use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelBridge, ParallelIterator};
use std::{fmt::Debug, iter::Sum};

pub struct WaveDecPlanner<T> {
    signal_length: usize,
    decomp_length: usize,
    levels: Vec<usize>,
    xforms: Vec<WaveletXForm1D<T>>,
    wavelet: Wavelet<T>,
    decomp_buffer: Vec<Complex<T>>,
    signal_buffer: Vec<Complex<T>>,
}


pub struct Decomp {
    lo_lo_lo:Array3<Complex32>,
    lo_lo_hi:Array3<Complex32>,
    lo_hi_lo:Array3<Complex32>,
    lo_hi_hi:Array3<Complex32>,
    hi_hi_hi:Array3<Complex32>,
    hi_hi_lo:Array3<Complex32>,
    hi_lo_hi:Array3<Complex32>,
    hi_lo_lo:Array3<Complex32>,
}


pub fn wavedec3(mut x:ArrayD<Complex32>,w:Wavelet<f32>) -> Vec<Array3<Complex32>> {

    let mut dims = x.shape().to_owned();

    for ax in 0..3 {

        let s_len = dims[ax];
        let wx = WaveletXForm1D::<f32>::new(s_len, w.filt_len());

        let mut new_dims = dims.clone();
        new_dims[ax] = wx.decomp_len();

        let mut result_buff = ArrayD::<Complex32>::zeros(new_dims.as_slice());

        x.lanes(Axis(ax)).into_iter().zip(result_buff.lanes_mut(Axis(ax))).par_bridge().for_each(|(x,mut y)|{
            let mut wx = wx.clone();
            let s = x.as_standard_layout().to_owned();
            let mut r = y.as_standard_layout().to_owned();
            wx.decompose(s.as_slice().unwrap(), w.lo_d(), w.hi_d(), r.as_slice_mut().unwrap());
            y.assign(&r);
        });

        x = result_buff;

        dims = new_dims;

    }

    let xd = dims[0]/2;
    let yd = dims[1]/2;
    let zd = dims[2]/2;

    let lo_lo_lo = x.slice(s![0..xd,0..yd,0..zd]);
    let lo_lo_hi = x.slice(s![0..xd,0..yd,zd..]);
    let lo_hi_lo = x.slice(s![0..xd,yd..,0..zd]);
    let lo_hi_hi = x.slice(s![0..xd,yd..,zd..]);

    let hi_hi_hi = x.slice(s![xd..,yd..,zd..]);
    let hi_hi_lo = x.slice(s![xd..,yd..,0..zd]);
    let hi_lo_hi = x.slice(s![xd..,0..yd,zd..]);
    let hi_lo_lo = x.slice(s![xd..,0..yd,0..zd]);


    vec![
        lo_lo_hi.to_owned(),
        lo_hi_lo.to_owned(),
        lo_hi_hi.to_owned(),
        hi_hi_hi.to_owned(),
        hi_hi_lo.to_owned(),
        hi_lo_hi.to_owned(),
        hi_lo_lo.to_owned(),
        lo_lo_lo.to_owned(),
    ]

}


pub fn waverec3(
    // llh:Array3<Complex32>,
    // lhl:Array3<Complex32>,
    // lhh:Array3<Complex32>,
    // hhh:Array3<Complex32>,
    // hhl:Array3<Complex32>,
    // hlh:Array3<Complex32>,
    // hll:Array3<Complex32>,
    // lll:Array3<Complex32>,
    subbands:&[Array3<Complex32>],
    target_dims:&[usize;3],
    w:Wavelet<f32>) -> Array3<Complex32> {

    // dec is a list of 7 arrays of the same shape

    let dims = subbands[0].shape().to_owned();

    let xd = dims[0];
    let yd = dims[1];
    let zd = dims[2];
    let coeff_dims = [xd,yd,zd];
    let mut input_dims = [xd*2,yd*2,zd*2];

    let mut input = Array3::<Complex32>::zeros(input_dims);
    
    
    input.slice_mut(s![0..xd,0..yd,zd..]).assign(&subbands[7]);
    input.slice_mut(s![0..xd,yd..,0..zd]).assign(&subbands[6]);
    input.slice_mut(s![0..xd,yd..,zd..]).assign(&subbands[5]);
    input.slice_mut(s![xd..,yd..,zd..]).assign(&subbands[4]);
    input.slice_mut(s![xd..,yd..,0..zd]).assign(&subbands[3]);
    input.slice_mut(s![xd..,0..yd,zd..]).assign(&subbands[2]);
    input.slice_mut(s![xd..,0..yd,0..zd]).assign(&subbands[1]);
    input.slice_mut(s![0..xd,0..yd,0..zd]).assign(&subbands[0]);


    for ax in 0..3 {
        let s_len = target_dims[ax];
        let wx = WaveletXForm1D::<f32>::new(s_len, w.filt_len());

        let mut new_dims = input_dims.clone();
        new_dims[ax] = s_len;

        let mut result_buff = Array3::<Complex32>::zeros(new_dims.clone());

        input.lanes(Axis(ax)).into_iter().zip(result_buff.lanes_mut(Axis(ax))).par_bridge().for_each(|(x,mut y)|{
            let mut wx = wx.clone();
            let s = x.as_standard_layout().to_owned();
            let approx = &s.as_slice().unwrap()[0..coeff_dims[ax]];
            let detail = &s.as_slice().unwrap()[coeff_dims[ax]..];
            let mut r = y.as_standard_layout().to_owned();
            wx.reconstruct(approx, detail, w.lo_r(), w.hi_r(), r.as_slice_mut().unwrap());
            y.assign(&r);
        });

        input = result_buff;
        input_dims = new_dims;

    }

    input

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

        let signal_energy = signal.par_iter().map(|x|x.norm_sqr()).sum::<T>();
        
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

        let result_energy = result.par_iter().map(|x|x.norm_sqr()).sum::<T>();

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
                    x[0],                              // number of approx coeffs
                    x[1],                              // signal length to reconstruct
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

        let decomp_energy = decomposed.par_iter().map(|x|x.norm_sqr()).sum::<T>();

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

        let result_energy = result.par_iter().map(|x|x.norm_sqr()).sum::<T>();

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
pub fn w_max_level(sig_len:usize,filt_len:usize) -> usize {
    if filt_len <= 1 {
        return 0
    }
    (sig_len as f32 / (filt_len as f32 - 1.)).log2() as usize
}