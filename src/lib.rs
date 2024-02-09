use crate::{
    dwt::{w_max_level, WaveDecPlanner, WaveRecPlanner, WaveletXForm1D}, wavelet::{Wavelet, WaveletType}
};
use num_complex::{Complex32, Complex64, ComplexFloat};
use num_traits::{One, Zero};
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use wavelet::WaveletFilter;
use std::{f64::consts::SQRT_2, time::Instant};

pub mod dwt;
mod utils;
pub mod wavelet;

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
