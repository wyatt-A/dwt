use std::{fmt::Debug, ops::{AddAssign, Mul, Range}, time::Instant};
use rustfft::{num_complex::{Complex, Complex32, Complex64, ComplexFloat}, num_traits::{float::FloatCore, zero, Float, FromPrimitive, Signed, Zero}, FftPlannerAvx};
use rayon::prelude::*;

fn main() {

    let n = 788;

    let ny = 480*480;

    let wavelet = Wavelet::<f64>::db2();
    let mut wavedec = WaveDecPlanner::new(n,4,wavelet);
    let mut waverec = WaveRecPlanner::new(&wavedec);

    let mut strides = vec![];
    let mut results = vec![];
    for _ in 0..ny {
        let x:Vec<Complex64> = (1..(n+1)).into_iter().map(|x| Complex64::new(x as f64,1.)).collect();
        strides.push(x);
        results.push(
            vec![Complex64::zero();wavedec.decomp_length]
        )
    }

    let now = Instant::now();


    strides.par_chunks_mut(1000).zip(results.par_chunks_mut(1000)).for_each(|(s,r)|{
        let mut wavedec = WaveDecPlanner::new(n,4,Wavelet::<f64>::db2());
        let mut waverec = WaveRecPlanner::new(&wavedec);
        for (stride,result) in s.iter_mut().zip(r.iter_mut()) {
            wavedec.process(stride,result);
            waverec.process(result,stride);
        }
    });

    // for (stride,result) in strides.iter_mut().zip(results.iter_mut()) {
    //     wavedec.process(stride,result);
    //     waverec.process(result,stride);
    // }

    let dur = now.elapsed();

    println!("{:#?}",strides.last().unwrap());
    println!("elapsed: {} ms", dur.as_millis());
}

#[derive(Clone)]
pub struct Wavelet<T> {
    lo_d:Vec<T>,
    hi_d:Vec<T>,
    lo_r:Vec<T>,
    hi_r:Vec<T>,
}
 impl<T> Wavelet<T>
 where T: FromPrimitive + Copy + Signed + Sync + Send + Debug + 'static
 {
    pub fn db2() -> Self {
        Self {
            
            lo_d: vec![
                T::from_f64(-0.1294095225509214464043594716713414527475833892822265625).unwrap(),
                T::from_f64(0.2241438680418573470287668669698177836835384368896484375).unwrap(),
                T::from_f64(0.836516303737468991386094785411842167377471923828125).unwrap(),
                T::from_f64(0.482962913144690253464119678028509952127933502197265625).unwrap()
            ],
            hi_d: vec![
                T::from_f64(-0.482962913144690253464119678028509952127933502197265625).unwrap(),
                T::from_f64(0.836516303737468991386094785411842167377471923828125).unwrap(),
                T::from_f64(-0.2241438680418573470287668669698177836835384368896484375).unwrap(),
                T::from_f64(-0.1294095225509214464043594716713414527475833892822265625).unwrap(),
            ],
            lo_r: vec![
                T::from_f64(0.482962913144690253464119678028509952127933502197265625).unwrap(),
                T::from_f64(0.836516303737468991386094785411842167377471923828125).unwrap(),
                T::from_f64(0.2241438680418573470287668669698177836835384368896484375).unwrap(),
                T::from_f64(-0.1294095225509214464043594716713414527475833892822265625).unwrap(),
            ],
            hi_r: vec![
                T::from_f64(-0.1294095225509214464043594716713414527475833892822265625).unwrap(),
                T::from_f64(-0.2241438680418573470287668669698177836835384368896484375).unwrap(),
                T::from_f64(0.836516303737468991386094785411842167377471923828125).unwrap(),
                T::from_f64(-0.482962913144690253464119678028509952127933502197265625).unwrap(),
            ],
        }
    }
 }


pub struct WaveDecPlanner<T> {
    signal_length:usize,
    decomp_length:usize,
    levels:Vec<usize>,
    xforms:Vec<WaveletXForm1D<T>>,
    wavelet: Wavelet<T>,
    decomp_buffer:Vec<Complex<T>>,
    signal_buffer:Vec<Complex<T>>,
}

impl<T> WaveDecPlanner<T>
where T: FromPrimitive + Copy + Signed + Sync + Send + Debug + 'static {
    pub fn new(signal_length:usize,n_levels:usize,wavelet:Wavelet<T>) -> Self {

        let mut levels = vec![0;n_levels+2];
        levels[n_levels+1] = signal_length;
        let mut xforms = vec![];
        let mut sig_len = signal_length;
        for level in 0..n_levels {
            let w = WaveletXForm1D::<T>::new(sig_len, wavelet.lo_d.len());
            levels[n_levels - level] = w.coeff_len;
            sig_len = w.coeff_len;
            xforms.push(w);
        }
        *levels.first_mut().unwrap() = xforms.last().unwrap().coeff_len;
    
        let mut decomp_len = xforms.iter().fold(0,|acc,x| acc + x.coeff_len);
        decomp_len += xforms.last().unwrap().coeff_len;
    
        let decomp = vec![Complex::<T>::zero();decomp_len];
        let signal = decomp.clone();

        Self {
            signal_length,
            decomp_length: decomp_len,
            levels: levels,
            xforms,
            wavelet,
            decomp_buffer:decomp,
            signal_buffer:signal,
        }
    }

    pub fn process(&mut self,signal:&[Complex<T>],result:&mut [Complex<T>]) {
        let mut stop = self.decomp_buffer.len();

        let lo_d = &self.wavelet.lo_d;
        let hi_d = &self.wavelet.hi_d;

        self.decomp_buffer[0..self.signal_length].copy_from_slice(&signal[0..self.signal_length]);
        let mut rl = 0;
        let mut ru = self.signal_length;
        for xform in self.xforms.iter_mut() {
            let start = stop - xform.decomp_len;
            self.signal_buffer[rl..ru].copy_from_slice(&self.decomp_buffer[rl..ru]);
            xform.decompose(&self.signal_buffer[rl..ru], lo_d, hi_d, &mut self.decomp_buffer[start..stop]);
            rl = start;
            ru = start + xform.coeff_len;
            stop -= xform.coeff_len;
        }

        //println!("decomp = {:#?}",self.decomp_buffer);

        result.copy_from_slice(&self.decomp_buffer);

        //self.decomp_buffer.clone()

    }   
}


pub struct WaveRecPlanner<T> {
    levels:Vec<usize>,
    xforms:Vec<(usize,usize,WaveletXForm1D<T>)>,
    signal_length:usize,
    signal_buffer:Vec<Complex<T>>,
    approx_buffer:Vec<Complex<T>>,
    wavelet:Wavelet<T>,
}

impl<T> WaveRecPlanner<T>
where T: FromPrimitive + Copy + Signed + Sync + Send + Debug + 'static {
    pub fn new(dec_planner:&WaveDecPlanner<T>) -> Self {

        let levels = dec_planner.levels.to_owned();
        let signal_length = dec_planner.signal_length;

        
        let signal = vec![Complex::<T>::zero();signal_length];
        let approx = vec![Complex::<T>::zero();signal_length];
        let xforms:Vec<_> = levels[1..].windows(2).map(|x| {
            (
                x[0], // number of approx coeffs
                x[1], // signal length to reconstruct
                WaveletXForm1D::<T>::new(x[1],4) // wavelet transform handler
            )
        }).collect();

        Self {
            levels:dec_planner.levels.to_owned(),
            xforms,
            signal_length,
            signal_buffer: signal,
            approx_buffer: approx,
            wavelet:dec_planner.wavelet.clone(),
        }
    }

    pub fn process(&mut self,decomposed:&[Complex<T>],result:&mut [Complex<T>]) {
        self.approx_buffer[0..self.levels[0]].copy_from_slice(&decomposed[0..self.levels[0]]);
        let lo_r = &self.wavelet.lo_r;
        let hi_r = &self.wavelet.hi_r;
        let mut start = self.levels[1];
        for(n_coeffs,sig_len,w) in self.xforms.iter_mut() {
            let detail = &decomposed[start..(start + *n_coeffs)];
            start += *n_coeffs;
            w.reconstruct(&self.approx_buffer[0..*n_coeffs], detail, lo_r, hi_r, &mut self.signal_buffer[0..*sig_len]);
            self.approx_buffer[0..*sig_len].copy_from_slice(&self.signal_buffer[0..*sig_len]);
        }
        //self.signal_buffer.clone()
        result.copy_from_slice(&self.signal_buffer);
    }

}


pub struct WaveletXForm1D<T> {
    /// length of filter coefficients
    filt_len:usize,
    /// length of signal
    sig_len:usize,
    /// length of decomposed signal
    decomp_len:usize,
    /// length of symmetric signal extension
    decomp_ext_len: usize,
    /// length of extended signal
    decomp_ext_result_len:usize,
    /// length of signal/filter convolution result
    decomp_conv_len:usize,
    /// length of detail/approx coefficents
    coeff_len:usize,
    /// length of reconstruction convolutions
    recon_conv_len:usize,
    /// lower index of convolution result center
    recon_conv_center_lidx:usize,
    /// upper index of convolution result center
    recon_conv_center_uidx:usize,

    decomp_conv_valid_lidx:usize,
    decomp_conv_valid_uidx:usize,
    
    decomp_scratch1:Vec<Complex<T>>,
    decomp_scratch2:Vec<Complex<T>>,
    decomp_scratch3:Vec<Complex<T>>,
    decomp_scratch4:Vec<Complex<T>>,

    recon_scratch1:Vec<Complex<T>>,
    recon_scratch2:Vec<Complex<T>>,
    recon_scratch3:Vec<Complex<T>>,
    recon_scratch4:Vec<Complex<T>>,

    recon_upsample_scratch:Vec<Complex<T>>

}

impl<T> WaveletXForm1D<T>
where T: FromPrimitive + Copy + Signed + Sync + Send + Debug + 'static
{
    pub fn new(sig_len:usize,filt_len:usize) -> Self {

        let decomp_ext_len = filt_len - 1;
        let decomp_len = 2*((filt_len + sig_len - 1)/2);
        let decomp_ext_result_len = 2*decomp_ext_len + sig_len;
        let decomp_conv_len = conv_len(decomp_ext_result_len, filt_len);
        let coeff_len = (filt_len + sig_len - 1)/2;

        let upsamp_len = coeff_len*2 - 1;
        let recon_conv_len = conv_len(upsamp_len, filt_len);
        let conv_center = _conv_center(recon_conv_len,sig_len);
        let decomp_conv_valid = _conv_valid(sig_len, filt_len);

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
            decomp_scratch1: vec![Complex::<T>::zero();decomp_conv_len],
            decomp_scratch2: vec![Complex::<T>::zero();decomp_conv_len],
            decomp_scratch3: vec![Complex::<T>::zero();decomp_conv_len],
            decomp_scratch4: vec![Complex::<T>::zero();decomp_conv_len],
            recon_scratch1: vec![Complex::<T>::zero();recon_conv_len],
            recon_scratch2: vec![Complex::<T>::zero();recon_conv_len],
            recon_scratch3: vec![Complex::<T>::zero();recon_conv_len],
            recon_scratch4: vec![Complex::<T>::zero();recon_conv_len],
            recon_upsample_scratch: vec![Complex::<T>::zero();upsamp_len],
        }
    }

    pub fn decomp_buffer(&self) -> Vec<Complex<T>> {
        vec![Complex::<T>::zero();self.decomp_len]
    }

    pub fn recon_buffer(&self,sig_len:usize) -> Vec<Complex<T>>{
        vec![Complex::<T>::zero();sig_len]
    }

    pub fn decompose(&mut self,signal:&[Complex<T>],lo_d:&[T],hi_d:&[T],decomp:&mut [Complex<T>]) {
        symm_ext(signal, self.decomp_ext_len, &mut self.decomp_scratch1);
        //conv1d(&self.decomp_scratch1[0..self.decomp_ext_result_len], lo_d,&mut self.decomp_scratch3,&mut self.decomp_scratch4, &mut self.decomp_scratch2);
        conv_direct(&self.decomp_scratch1[0..self.decomp_ext_result_len], lo_d, &mut self.decomp_scratch2);
        downsample2(&self.decomp_scratch2[conv_valid(self.decomp_ext_result_len, self.filt_len)], &mut decomp[0..self.coeff_len]);
        //conv1d(&self.decomp_scratch1[0..self.decomp_ext_result_len], hi_d,&mut self.decomp_scratch3,&mut self.decomp_scratch4, &mut self.decomp_scratch2);
        conv_direct(&self.decomp_scratch1[0..self.decomp_ext_result_len], hi_d, &mut self.decomp_scratch2);
        downsample2(&self.decomp_scratch2[conv_valid(self.decomp_ext_result_len, self.filt_len)], &mut decomp[self.coeff_len..]);
    }

    pub fn reconstruct(&mut self,approx:&[Complex<T>],detail:&[Complex<T>],lo_r:&[T],hi_r:&[T],signal:&mut [Complex<T>]) {

        let conv_center = _conv_center(self.recon_conv_len, signal.len());

        upsample_odd(approx,&mut self.recon_upsample_scratch);
        //conv1d(&self.recon_upsample_scratch, lo_r, &mut self.recon_scratch3, &mut self.recon_scratch4, &mut self.recon_scratch1);
        conv_direct(&self.recon_upsample_scratch, lo_r , &mut self.recon_scratch1);
        upsample_odd(detail,&mut self.recon_upsample_scratch);
        //conv1d(&self.recon_upsample_scratch, &hi_r, &mut self.recon_scratch3, &mut self.recon_scratch4, &mut self.recon_scratch2);
        conv_direct(&self.recon_upsample_scratch, &hi_r, &mut self.recon_scratch2);
        let a = &self.recon_scratch1[conv_center.0..conv_center.1];
        let d = &self.recon_scratch2[conv_center.0..conv_center.1];
        signal.iter_mut().enumerate().for_each(|(idx,x)|{
            *x = a[idx] + d[idx];
        });
    }
}   

pub fn _wavelet_decomp<T>(w:&mut WaveletXForm1D<T>,signal:&[Complex<T>],lo_d:&[T],hi_d:&[T],decomp:&mut [Complex<T>])
where T: FromPrimitive + Copy + Signed + Sync + Send + Debug + 'static {
    symm_ext(signal, w.decomp_ext_len, &mut w.decomp_scratch1);
    conv1d(&w.decomp_scratch1[0..w.decomp_ext_result_len], lo_d,&mut w.decomp_scratch3,&mut w.decomp_scratch4, &mut w.decomp_scratch2);
    downsample2(&w.decomp_scratch2[conv_valid(w.decomp_ext_result_len, w.filt_len)], &mut decomp[0..w.coeff_len]);
    conv1d(&w.decomp_scratch1[0..w.decomp_ext_result_len], hi_d,&mut w.decomp_scratch3,&mut w.decomp_scratch4, &mut w.decomp_scratch2);
    downsample2(&w.decomp_scratch2[conv_valid(w.decomp_ext_result_len, w.filt_len)], &mut decomp[w.coeff_len..]);
}


pub fn wavelet_decomp<T>(signal:&[Complex<T>],lo_d:&[T],hi_d:&[T],decomp:&mut [Complex<T>])
where T: FromPrimitive + Copy + Signed + Sync + Send + Debug + 'static {



    let filt_len = lo_d.len();
    let sig_len = signal.len();
    let ext_len = filt_len - 1;
    let ext_result_len = 2*ext_len + sig_len;
    let conv_result_len = conv_len(ext_result_len, filt_len);
    let coeff_len = (filt_len + sig_len - 1)/2;


    let mut scratch_buff1 = vec![Complex::<T>::zero();conv_result_len];
    let mut scratch_buff2 = scratch_buff1.clone();

    let mut scratch_buff3 = scratch_buff1.clone();
    let mut scratch_buff4 = scratch_buff1.clone();


    symm_ext(signal, ext_len, &mut scratch_buff1);

    conv1d(&scratch_buff1[0..ext_result_len], lo_d,&mut scratch_buff3,&mut scratch_buff4, &mut scratch_buff2);
    downsample2(&scratch_buff2[conv_valid(ext_result_len, filt_len)], &mut decomp[0..coeff_len]);

    conv1d(&scratch_buff1[0..ext_result_len], hi_d,&mut scratch_buff3,&mut scratch_buff4, &mut scratch_buff2);
    downsample2(&scratch_buff2[conv_valid(ext_result_len, filt_len)], &mut decomp[coeff_len..]);
}

pub fn _wavelet_recon<T>(w:&mut WaveletXForm1D<T>,decomp:&[Complex<T>],lo_r:&[T],hi_r:&[T],signal:&mut [Complex<T>])
where T: FromPrimitive + Copy + Signed + Sync + Send + Debug + 'static {
    upsample_odd(&decomp[0..w.coeff_len],&mut w.recon_upsample_scratch);
    conv1d(&w.recon_upsample_scratch, lo_r, &mut w.recon_scratch3, &mut w.recon_scratch4, &mut w.recon_scratch1);
    upsample_odd(&decomp[w.coeff_len..],&mut w.recon_upsample_scratch);
    conv1d(&w.recon_upsample_scratch, &hi_r, &mut w.recon_scratch3, &mut w.recon_scratch4, &mut w.recon_scratch2);
    let a = &w.recon_scratch1[w.recon_conv_center_lidx..w.recon_conv_center_uidx];
    let d = &w.recon_scratch2[w.recon_conv_center_lidx..w.recon_conv_center_uidx];
    signal.iter_mut().enumerate().for_each(|(idx,x)|{
        *x = a[idx] + d[idx];
    });

}


pub fn wavelet_recon<T>(decomp:&[Complex<T>],lo_r:&[T],hi_r:&[T],signal:&mut [Complex<T>])
where T: FromPrimitive + Copy + Signed + Sync + Send + Debug + 'static {

    let filt_len = lo_r.len();
    let coeff_len = decomp.len()/2;

    let sig_len = signal.len();

    let upsamp_len = coeff_len*2 - 1;

    let conv_len = conv_len(upsamp_len, filt_len);

    let conv_center = conv_center(conv_len,sig_len);

    let mut scratch_buff1 = vec![Complex::<T>::zero();upsamp_len];
    let mut scratch_buff2 = vec![Complex::<T>::zero();conv_len];
    let mut scratch_buff3 = scratch_buff2.clone();

    let mut conv_scratch1 = scratch_buff2.clone();
    let mut conv_scratch2 = scratch_buff2.clone();

    upsample_odd(&decomp[0..coeff_len],&mut scratch_buff1);
    conv1d(&scratch_buff1, lo_r, &mut conv_scratch1, &mut conv_scratch2, &mut scratch_buff2);

    upsample_odd(&decomp[coeff_len..],&mut scratch_buff1);
    conv1d(&scratch_buff1, &hi_r, &mut conv_scratch1, &mut conv_scratch2, &mut scratch_buff3);

    let a = &scratch_buff2[conv_center.clone()];
    let d = &scratch_buff3[conv_center];

    signal.iter_mut().enumerate().for_each(|(idx,x)|{
        *x = a[idx] + d[idx];
    });

}


fn conv_direct<T>(input: &[Complex<T>], kernel: &[T], result: &mut [Complex<T>])
where T: FromPrimitive + Copy + Signed + Sync + Send + Debug + 'static {
    
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



fn downsample2<T>(sig: &[Complex<T>], downsampled: &mut [Complex<T>])
where T: FromPrimitive + Copy + Signed + Sync + Send + Debug + 'static {
    downsampled.iter_mut().enumerate().for_each(|(idx,r)|{
        *r = sig[2*idx + 1]
    })
}


fn upsample_odd<T>(sig: &[Complex<T>], upsampled: &mut [Complex<T>])
where T: FromPrimitive + Copy + Signed + Sync + Send + Debug + 'static {
    upsampled.iter_mut().for_each(|x| *x = Complex::<T>::zero());
    sig.iter().enumerate().for_each(|(idx,x)|{
        upsampled[2*idx] = *x;
    })
}


fn symm_ext<T>(sig: &[Complex<T>], a: usize, oup: &mut [Complex<T>])
where T: FromPrimitive + Copy + Signed + Sync + Send + Debug + 'static
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

fn per_ext<T>(sig: &[Complex<T>], a: usize, oup: &mut [Complex<T>])
where T: FromPrimitive + Copy + Signed + Sync + Send + Debug + 'static{
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

fn conv_valid(sig_len:usize,filt_len:usize) -> Range<usize> {
    if filt_len < 1 {
        panic!("filter length must be greater than 0");
    }
    (filt_len - 1)..sig_len 
}
fn _conv_valid(sig_len:usize,filt_len:usize) -> (usize,usize) {
    if filt_len < 1 {
        panic!("filter length must be greater than 0");
    }
    (filt_len - 1,sig_len)
}

fn conv_center(sig_len:usize,center_len:usize) -> Range<usize> {

    let f = (sig_len - center_len)/2;

    f..(f + center_len)

}

fn _conv_center(sig_len:usize,center_len:usize) -> (usize,usize) {
    let f = (sig_len - center_len)/2;
    (f,f+center_len)
}

pub fn conv1d<T>(signal:&[Complex<T>],filter:&[T],s_buff1:&mut [Complex<T>],s_buff2:&mut [Complex<T>],result:&mut [Complex<T>])
where T: FromPrimitive + Copy + Signed + Sync + Send + Debug + 'static
{

    let n = result.len();
    
    s_buff1.iter_mut().for_each(|x| *x = Complex::<T>::zero());
    s_buff2.iter_mut().for_each(|x| *x = Complex::<T>::zero());

    s_buff1.iter_mut().zip(signal.iter()).for_each(|(x,s)|{
        *x = *s;
    });

    s_buff2.iter_mut().zip(filter.iter()).for_each(|(x,f)|{
        *x = Complex::<T>::new(*f,T::zero());
    });

    let mut fp = FftPlannerAvx::new().unwrap();
    let fft = fp.plan_fft_forward(n);
    let ifft = fp.plan_fft_inverse(n);
    fft.process(s_buff1);
    fft.process(s_buff2);

    s_buff1.into_iter().zip(s_buff2.into_iter()).zip(result.iter_mut()).for_each(|((x,y),r)|{
        *r = *x * *y;
    });

    ifft.process(result);


    let scale = T::one()/T::from_usize(n).unwrap();

    result.iter_mut().for_each(|x| {
        *x = *x * scale;
    })

}

pub fn conv_len(sig_len:usize,filt_len:usize) -> usize {
    sig_len + filt_len - 1
}

#[test]
fn test2(){
    
    let n = 20;

    let x:Vec<Complex64> = (1..(n+1)).into_iter().map(|x| Complex64::new(x as f64,0.)).collect();

    let lo_d = [
        -0.1294095225509214464043594716713414527475833892822265625,
        0.2241438680418573470287668669698177836835384368896484375,
        0.836516303737468991386094785411842167377471923828125,
        0.482962913144690253464119678028509952127933502197265625
    ];

    let hi_d = [
        -0.482962913144690253464119678028509952127933502197265625000000,
        0.836516303737468991386094785411842167377471923828125000000000,
        -0.224143868041857347028766866969817783683538436889648437500000,
        -0.129409522550921446404359471671341452747583389282226562500000,
    ];

    let lo_r = [
        0.482962913144690253464119678028509952127933502197265625000000,
        0.836516303737468991386094785411842167377471923828125000000000,
        0.224143868041857347028766866969817783683538436889648437500000,
        -0.129409522550921446404359471671341452747583389282226562500000,
    ];

    let hi_r = [
        -0.129409522550921446404359471671341452747583389282226562500000,
        -0.224143868041857347028766866969817783683538436889648437500000,
        0.836516303737468991386094785411842167377471923828125000000000,
        -0.482962913144690253464119678028509952127933502197265625000000
    ];


    println!("len x = {}",x.len());

    
    let j = 3;
    let mut levels = vec![0;j+2];
    levels[j+1] = x.len();
    let mut xforms = vec![];
    let mut sig_len = x.len();
    for level in 0..j {
        let w = WaveletXForm1D::<f64>::new(sig_len, lo_d.len());
        levels[j - level] = w.coeff_len;
        sig_len = w.coeff_len;
        xforms.push(w);
    }
    *levels.first_mut().unwrap() = xforms.last().unwrap().coeff_len;


    let mut decomp_len = xforms.iter().fold(0,|acc,x| acc + x.coeff_len);
    decomp_len += xforms.last().unwrap().coeff_len;

    println!("decomp_len = {}",decomp_len);
    //let mut signal = x.to_owned();
    let mut stop = decomp_len;
    let mut decomp = vec![Complex64::zero();decomp_len];

    let mut signal = decomp.clone();
    decomp[0..x.len()].copy_from_slice(&x);
    let mut rl = 0;
    let mut ru = x.len();
    for xform in xforms.iter_mut() {
        let start = stop - xform.decomp_len;
        signal[rl..ru].copy_from_slice(&decomp[rl..ru]);
        xform.decompose(&signal[rl..ru], &lo_d, &hi_d, &mut decomp[start..stop]);
        rl = start;
        ru = start + xform.coeff_len;
        stop -= xform.coeff_len;
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
            WaveletXForm1D::new(x[1],4) // wavelet transform handler
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