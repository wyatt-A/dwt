use std::{fmt::Debug, ops::{AddAssign, Mul, Range}};

use rustfft::{num_complex::{Complex, Complex32, Complex64, ComplexFloat}, num_traits::{float::FloatCore, zero, Float, FromPrimitive, Signed, Zero}, FftPlannerAvx};


fn main() {
    println!("Hello, world!");
                
    //let x = vec![Complex32::new(0.9597,0.),Complex32::new(0.3404,0.),Complex32::new(0.5853,0.),Complex32::new(0.2238,0.)];

    let x:Vec<Complex32> = (1..21).into_iter().map(|x| Complex32::new(x as f32,0.)).collect();

    let lo_d = [-0.129409522550921,0.224143868041857,0.836516303737469,0.482962913144690];
    let hi_d = [-0.482962913144690,0.836516303737469,-0.224143868041857,-0.129409522550921];
    let lo_r = [0.482962913144690,0.836516303737469,0.224143868041857,-0.129409522550921];
    let hi_r = [-0.129409522550921,-0.224143868041857,0.836516303737469,-0.482962913144690];


    let n_levels = 4;

    let  mut xforms = vec![];
    let mut sig_len = x.len();
    for _ in 0..n_levels {
        let w = WaveletXForm1D::new(sig_len, lo_d.len());
        sig_len = w.coeff_len;
        xforms.push(w);
    }

    let mut decomp_len = xforms.iter().fold(0,|acc,x| acc + x.coeff_len);
    decomp_len += xforms.last().unwrap().coeff_len;


    println!("decomp_len = {}",decomp_len);

    let mut signal = x.to_owned();
    let mut stop = decomp_len;
    let mut decomp = vec![Complex32::zero();decomp_len];

    for xform in xforms.iter_mut() {
        //let mut d = xform.decomp_buffer();
        let start = stop - xform.decomp_len;
        xform.decompose(&signal, &lo_d, &hi_d, &mut decomp[start..stop]);
        // copy the approximation coeffs
        signal = decomp[start..(start + xform.coeff_len)].to_owned();
        stop -= xform.coeff_len;
    }

    println!("decomp len = {}",decomp.len());
    println!("decomp: {:#?}",decomp);

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

    pub fn recon_buffer(&self) -> Vec<Complex<T>>{
        vec![Complex::<T>::zero();self.sig_len]
    }

    pub fn decompose(&mut self,signal:&[Complex<T>],lo_d:&[T],hi_d:&[T],decomp:&mut [Complex<T>]) {
        symm_ext(signal, self.decomp_ext_len, &mut self.decomp_scratch1);
        conv1d(&self.decomp_scratch1[0..self.decomp_ext_result_len], lo_d,&mut self.decomp_scratch3,&mut self.decomp_scratch4, &mut self.decomp_scratch2);
        downsample2(&self.decomp_scratch2[conv_valid(self.decomp_ext_result_len, self.filt_len)], &mut decomp[0..self.coeff_len]);
        conv1d(&self.decomp_scratch1[0..self.decomp_ext_result_len], hi_d,&mut self.decomp_scratch3,&mut self.decomp_scratch4, &mut self.decomp_scratch2);
        downsample2(&self.decomp_scratch2[conv_valid(self.decomp_ext_result_len, self.filt_len)], &mut decomp[self.coeff_len..]);
    }

    pub fn reconstruct(&mut self,decomp:&[Complex<T>],lo_r:&[T],hi_r:&[T],signal:&mut [Complex<T>]) {
        upsample_odd(&decomp[0..self.coeff_len],&mut self.recon_upsample_scratch);
        conv1d(&self.recon_upsample_scratch, lo_r, &mut self.recon_scratch3, &mut self.recon_scratch4, &mut self.recon_scratch1);
        upsample_odd(&decomp[self.coeff_len..],&mut self.recon_upsample_scratch);
        conv1d(&self.recon_upsample_scratch, &hi_r, &mut self.recon_scratch3, &mut self.recon_scratch4, &mut self.recon_scratch2);
        let a = &self.recon_scratch1[self.recon_conv_center_lidx..self.recon_conv_center_uidx];
        let d = &self.recon_scratch2[self.recon_conv_center_lidx..self.recon_conv_center_uidx];
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