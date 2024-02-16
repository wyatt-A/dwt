mod utils;
pub mod wavelet;

use ndarray::{s, ArrayD, Axis};
use num_complex::{Complex, Complex32};
use num_traits::{Float, FromPrimitive, Signed, Zero};
use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelBridge, ParallelIterator};
use utils::*;
use wavelet::*;
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

/// single-level 3D wavelet transform
pub fn dwt3(mut x:ArrayD<Complex32>,w:Wavelet<f32>) -> ArrayD<Complex32> {

    let mut dims = x.shape().to_owned();

    if dims.len() != 3 {
        println!("array must have 3 dimensions for 3D wavelet transform!");
        assert!(dims.len() == 3);
    }

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
    x
}

/// single level inverse 3D wavelet transform. Accepts an array of subbands and returns the original
/// signal with size of specified target dims. Target dims must be equal to the original signals size
/// for an accurate reconstruction.
pub fn idwt3(mut x:ArrayD<Complex32>,w:Wavelet<f32>,target_dims:&[usize]) -> ArrayD<Complex32> {

    let mut dims = x.shape().to_owned();

    if dims.len() != 3 {
        println!("array must have 3 dimensions for 3D wavelet transform!");
        assert!(dims.len() == 3);
    }

    let coeff_dims:Vec<_> = dims.iter().map(|d| d/2).collect();

    for ax in 0..3 {
        let s_len = target_dims[ax];
        let wx = WaveletXForm1D::<f32>::new(s_len, w.filt_len());

        let mut new_dims = dims.clone();
        new_dims[ax] = s_len;

        let mut result_buff = ArrayD::<Complex32>::zeros(new_dims.clone());

        x.lanes(Axis(ax)).into_iter().zip(result_buff.lanes_mut(Axis(ax))).par_bridge().for_each(|(x,mut y)|{
            let mut wx = wx.clone();
            let s = x.as_standard_layout().to_owned();
            let approx = &s.as_slice().unwrap()[0..coeff_dims[ax]];
            let detail = &s.as_slice().unwrap()[coeff_dims[ax]..];
            let mut r = y.as_standard_layout().to_owned();
            wx.reconstruct(approx, detail, w.lo_r(), w.hi_r(), r.as_slice_mut().unwrap());
            y.assign(&r);
        });

        x = result_buff;
        dims = new_dims;

    }

    x
}

/// single-level 3D wavelet transform. Returns a vec of subbands in order:
/// LLH, LHL, LHH, HHH, HHL, HLH, HLL, LLL
pub fn wavedec3_single_level(x:ArrayD<Complex32>,w:Wavelet<f32>) -> Vec<ArrayD<Complex32>> {
    
    let x = dwt3(x, w);

    let dims = x.shape();

    // bandsizes are half dimmensions of result
    let xb = dims[0]/2;
    let yb = dims[1]/2;
    let zb = dims[2]/2;

    // slice into bands and return owned sub arrays
    let lo_lo_lo = x.slice(s![0..xb,0..yb,0..zb]);
    let lo_lo_hi = x.slice(s![0..xb,0..yb,zb..]);
    let lo_hi_lo = x.slice(s![0..xb,yb..,0..zb]);
    let lo_hi_hi = x.slice(s![0..xb,yb..,zb..]);

    let hi_hi_hi = x.slice(s![xb..,yb..,zb..]);
    let hi_hi_lo = x.slice(s![xb..,yb..,0..zb]);
    let hi_lo_hi = x.slice(s![xb..,0..yb,zb..]);
    let hi_lo_lo = x.slice(s![xb..,0..yb,0..zb]);

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
pub fn waverec3_single_level(subbands:&[ArrayD<Complex32>],w:Wavelet<f32>,target_dims:&[usize]) -> ArrayD<Complex32> {

    if subbands.len() != 8 {
        println!("the number of subbands must be 8");
        assert!(subbands.len() == 8);
    }

    let dims = subbands[0].shape().to_owned();

    for i in 1..8 {
        assert_eq!(&dims,subbands[i].shape(),"subbands have inconsistent shapes!");
    }

    // extract bandsizes for array assignment
    let xb = dims[0];
    let yb = dims[1];
    let zb = dims[2];

    let block_dims:Vec<_> = dims.iter().map(|d| d*2).collect();

    let mut x = ArrayD::<Complex32>::zeros(block_dims);
    
    x.slice_mut(s![0..xb,0..yb,zb..]).assign(&subbands[7]);
    x.slice_mut(s![0..xb,yb..,0..zb]).assign(&subbands[6]);
    x.slice_mut(s![0..xb,yb..,zb..]).assign(&subbands[5]);
    x.slice_mut(s![xb..,yb..,zb..]).assign(&subbands[4]);
    x.slice_mut(s![xb..,yb..,0..zb]).assign(&subbands[3]);
    x.slice_mut(s![xb..,0..yb,zb..]).assign(&subbands[2]);
    x.slice_mut(s![xb..,0..yb,0..zb]).assign(&subbands[1]);
    x.slice_mut(s![0..xb,0..yb,0..zb]).assign(&subbands[0]);

    idwt3(x, w, target_dims)

}

pub struct WaveDec3 {
    subbands:Vec<ArrayD<Complex32>>,
    signal_dims_per_level:Vec<Vec<usize>>,
    wavelet:Wavelet<f32>
}

pub fn wavedec3(x:ArrayD<Complex32>,w:Wavelet<f32>,num_levels:usize) -> WaveDec3 {

    assert!(num_levels != 0,"num_levels must be greater than 0!");

    let mut signal_dims = vec![];
    signal_dims.push(x.shape().to_owned());

    // returns vector of subbands (7 per level, except for the last level)
    let mut x = wavedec3_single_level(x, w.clone());

    // for each level, pop the LLL subband, decompose it, then push those subbands to the stack
    for _ in 1..num_levels {
        let input = x.pop().unwrap().into_dyn();
        signal_dims.push(input.shape().to_owned());
        let mut y = wavedec3_single_level(input,w.clone());
        x.append(&mut y);
    }

    WaveDec3 {
        subbands: x,
        signal_dims_per_level: signal_dims,
        wavelet:w
    }

}

/// Multi-level 3D wavelet reconstruction from a previous deconstruction
pub fn waverec3(mut dec:WaveDec3) -> ArrayD<Complex32> {

    dec.signal_dims_per_level.reverse();
    for s in dec.signal_dims_per_level {
        let subbands:Vec<_> = (0..8).map(|_| dec.subbands.pop().unwrap() ).collect();
        let rec = waverec3_single_level(&subbands, dec.wavelet.clone(), &s );
        dec.subbands.push(rec)
    }

    let rec = dec.subbands.pop().unwrap();
    assert!(dec.subbands.len() == 0,"not all subbands were reconstructed!");
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

mod tests {
    use std::time::Instant;

    use num_complex::{Complex64, ComplexFloat};
    use num_traits::One;
    use rayon::{iter::IndexedParallelIterator, slice::ParallelSliceMut};

    use super::*;

    #[test]
    fn test() {
        let n = 788;
    
        let ny = 480 * 480;
    
        //let wavelet = Wavelet::<f32>::new(WaveletType::Daubechies2);
        let wavedec = WaveDecPlanner::<f32>::new(n, 4, Wavelet::new(WaveletType::Daubechies2));
        let mut strides = vec![];
        let mut results = vec![];
        for _ in 0..ny {
            let x: Vec<Complex32> = (1..(n + 1))
                .into_iter()
                .map(|x| Complex32::new(x as f32,0.))
                .collect();
            strides.push(x);
            results.push(vec![Complex32::zero(); wavedec.decomp_len()])
        }
    
        let now = Instant::now();
    
        strides
            .par_chunks_mut(100)
            .zip(results.par_chunks_mut(100))
            .for_each(|(s, r)| {
                let mut wavedec = WaveDecPlanner::new(n, 4, Wavelet::new(WaveletType::Daubechies2));
                let mut waverec = WaveRecPlanner::new(&wavedec);
                for (stride, result) in s.iter_mut().zip(r.iter_mut()) {
                    wavedec.process(stride, result);
                    waverec.process(result, stride);
                }
            });
    
        let dur = now.elapsed();
    
        println!("{:#?}", strides.last().unwrap());
        println!("elapsed: {} ms", dur.as_millis());
    }
    
    #[test]
    fn test3(){
        let n = 788;
        //let x:Vec<Complex64> = (1..(n+1)).into_iter().map(|x| Complex64::new(x as f64,0.)).collect();
        let x = vec![Complex64::one();n];
    
        // measure energy of x
        let x_energ = x.par_iter().map(|x|x.norm_sqr()).sum::<f64>();
    
        let wavelet = Wavelet::<f64>::new(WaveletType::Daubechies10);
    
        let n_levels = w_max_level(x.len(), wavelet.filt_len());
        //let n_levels = 1;
        println!("max levels: {}",n_levels);
    
        let mut wavedec = WaveDecPlanner::<f64>::new(n, n_levels, wavelet);
        let mut waverec = WaveRecPlanner::<f64>::new(&wavedec);
    
        let mut result = vec![Complex64::zero();wavedec.decomp_len()];
        let mut recon = vec![Complex64::zero();x.len()];
    
        wavedec.process(&x, &mut result);
    
        // measure energy of result
        //let w_energ = result.par_iter().map(|x|x.norm_sqr()).sum::<f64>();
    
        //let scale = (x_energ / w_energ).sqrt();
    
        //result.iter_mut().for_each(|x| *x = *x * scale);
    
        let result_energ = result.par_iter().map(|x|x.norm_sqr()).sum::<f64>();
    
        waverec.process(&result, &mut recon);
    
        let max_err = recon.iter().zip(x.iter()).map(|(r,x)| (*x - *r).abs()).max_by(|a,b|a.partial_cmp(&b).unwrap()).unwrap();
    
        let recon_energy = recon.par_iter().map(|x|x.norm_sqr()).sum::<f64>();
    
    
        println!("max error: {}",max_err);
    
        println!("x energy: {}",x_energ);
        println!("decomp energy: {}",result_energ);
        println!("recon energy: {}",recon_energy);
    
    }
    
    
    #[test]
    fn test_mlevel() {
        let n = 20;
        let mut x:Vec<Complex64> = (1..(n+1)).into_iter().map(|x| Complex64::new(x as f64,0.)).collect();
        let wavelet = Wavelet::<f64>::new(WaveletType::Daubechies4);
    
        let filt_len = wavelet.filt_len();
        let lo_d = wavelet.lo_d();
        let hi_d = wavelet.hi_d();
    
        let mut sig_len = n;
    
        let mut decomp_buff = vec![Complex64::zero();200];
        let mut approx_buff = vec![Complex64::zero();200];
        let mut decomp_result = vec![];
        let mut sig = x.as_mut_slice();
    
        let mut levels = vec![];
    
        for _ in 0..2 {
            let mut xform = WaveletXForm1D::new(sig_len,filt_len);
            let d_len = xform.decomp_len();
    
            levels.push((sig_len,d_len));
    
            //println!("decomp_len = {}",d_len);
            xform.decompose(sig, lo_d, hi_d, &mut decomp_buff[0..d_len]);
            approx_buff[0..d_len/2].copy_from_slice(&decomp_buff[0..d_len/2]);
            decomp_buff[d_len/2..d_len].reverse();
            decomp_result.extend_from_slice(&decomp_buff[d_len/2..d_len]);
            sig = &mut approx_buff[0..d_len/2];
            sig_len = sig.len();
        }
        sig.reverse();
        decomp_result.extend_from_slice(sig);
        decomp_result.reverse();
        println!("decomp = {:#?}",decomp_result);
    
        //let mut recon_result = vec![];
        let mut recon_buffer = vec![Complex64::zero();200];
    
        let lo_r = wavelet.lo_r();
        let hi_r = wavelet.hi_r();
    
        println!("{:#?}",levels);
    
        let mut recon_buff = vec![Complex64::zero();200];
    
        let (mut sig_len,mut d_len) = levels.last().unwrap();
    
        let mut approx = &decomp_buff[0..d_len/2];
        let mut detail = &decomp_buff[d_len/2..d_len];
    
        let mut xform = WaveletXForm1D::new(sig_len,filt_len);
    
        xform.reconstruct(approx, detail, lo_r, hi_r, &mut recon_buff[0..sig_len]);
    
        println!("{:#?}",&recon_buff[0..sig_len]);
    
    }
    
    
    #[test]
    fn test4() {
        let n = 20;
        let x:Vec<Complex64> = (1..(n+1)).into_iter().map(|x| Complex64::new(x as f64,0.)).collect();
        let wavelet = Wavelet::<f64>::new(WaveletType::Daubechies5);
    
        let mut w = WaveletXForm1D::new(x.len(), wavelet.lo_d().len());
    
        let mut d = w.decomp_buffer();
    
        w.decompose(&x, wavelet.lo_d(), wavelet.hi_d(), &mut d);
    
        let mut recon = w.recon_buffer(x.len());
    
        w.reconstruct(&d[0..d.len()/2], &d[d.len()/2 ..], wavelet.lo_r(), wavelet.hi_r(),&mut recon);
    
        println!("recon = {:#?}",recon);
    
    }
    

    #[test]
    fn test2(){
    
        let n = 20;
    
        let x:Vec<Complex64> = (1..(n+1)).into_iter().map(|x| Complex64::new(x as f64,0.)).collect();
    
        let wavelet = Wavelet::<f64>::new(WaveletType::Daubechies4);
    
        let filt_len = wavelet.filt_len();
        let lo_d = wavelet.lo_d();
        let hi_d = wavelet.hi_d();
        let lo_r = wavelet.lo_r();
        let hi_r = wavelet.hi_r();
    
    
        println!("len x = {}",x.len());
    
        let j = 3;
        let mut levels = vec![0;j+2];
        levels[j+1] = x.len();
        let mut xforms = vec![];
        let mut sig_len = x.len();
        for level in 0..j {
            let w = WaveletXForm1D::<f64>::new(sig_len, filt_len);
            levels[j - level] = w.coeff_len();
            sig_len = w.coeff_len();
            xforms.push(w);
        }
        *levels.first_mut().unwrap() = xforms.last().unwrap().coeff_len();
    
        let mut decomp_len = xforms.iter().fold(0,|acc,x| acc + x.coeff_len());
        decomp_len += xforms.last().unwrap().coeff_len();
    
        println!("decomp_len = {}",decomp_len);
        //let mut signal = x.to_owned();
        let mut stop = decomp_len;
        let mut decomp = vec![Complex64::zero();decomp_len];
    
        let mut signal = decomp.clone();
        decomp[0..x.len()].copy_from_slice(&x);
        let mut rl = 0;
        let mut ru = x.len();
        for xform in xforms.iter_mut() {
            let start = stop - xform.decomp_len();
            signal[rl..ru].copy_from_slice(&decomp[rl..ru]);
            xform.decompose(&signal[rl..ru], &lo_d, &hi_d, &mut decomp[start..stop]);
            rl = start;
            ru = start + xform.coeff_len();
            stop -= xform.coeff_len();
        }
    
        println!("decomp = {:#?}",decomp);
    
        // initialization
        // inputs are decomp and levels
        let mut start = levels[1];
        let mut signal = vec![Complex64::zero();x.len()];
        let mut approx = vec![Complex64::zero();x.len()];
        let mut xforms:Vec<_> = levels[1..].windows(2).map(|x| {
            (
                x[0], // number of approx coeffs
                x[1], // signal length to reconstruct
                WaveletXForm1D::new(x[1],filt_len) // wavelet transform handler
            )
        }).collect();
    
        // the following can be repeated with different decomp arrays without new allocations
        approx[0..levels[0]].copy_from_slice(&decomp[0..levels[0]]);
    
        for(n_coeffs,sig_len,w) in xforms.iter_mut() {
            let detail = &decomp[start..(start + *n_coeffs)];
            start += *n_coeffs;
            w.reconstruct(&approx[0..*n_coeffs], detail, &lo_r, &hi_r, &mut signal[0..*sig_len]);
            approx[0..*sig_len].copy_from_slice(&signal[0..*sig_len]);
            //approx = signal[0..sig_len].to_owned();
            println!("signal = {:#?}",signal);
        }
    }    
}