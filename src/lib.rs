use crate::{
    dwt::{WaveDecPlanner, WaveRecPlanner, WaveletXForm1D},
    wavelet::{Wavelet, WaveletType},
};
use num_complex::Complex32;
use num_traits::Zero;
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use std::time::Instant;

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
            .map(|x| Complex32::new(x as f32, 1.))
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

// #[test]
// fn test2(){

//     let n = 20;

//     let x:Vec<Complex64> = (1..(n+1)).into_iter().map(|x| Complex64::new(x as f64,0.)).collect();

//     let lo_d = [
//         -0.1294095225509214464043594716713414527475833892822265625,
//         0.2241438680418573470287668669698177836835384368896484375,
//         0.836516303737468991386094785411842167377471923828125,
//         0.482962913144690253464119678028509952127933502197265625
//     ];

//     let hi_d = [
//         -0.482962913144690253464119678028509952127933502197265625000000,
//         0.836516303737468991386094785411842167377471923828125000000000,
//         -0.224143868041857347028766866969817783683538436889648437500000,
//         -0.129409522550921446404359471671341452747583389282226562500000,
//     ];

//     let lo_r = [
//         0.482962913144690253464119678028509952127933502197265625000000,
//         0.836516303737468991386094785411842167377471923828125000000000,
//         0.224143868041857347028766866969817783683538436889648437500000,
//         -0.129409522550921446404359471671341452747583389282226562500000,
//     ];

//     let hi_r = [
//         -0.129409522550921446404359471671341452747583389282226562500000,
//         -0.224143868041857347028766866969817783683538436889648437500000,
//         0.836516303737468991386094785411842167377471923828125000000000,
//         -0.482962913144690253464119678028509952127933502197265625000000
//     ];

//     println!("len x = {}",x.len());

//     let j = 3;
//     let mut levels = vec![0;j+2];
//     levels[j+1] = x.len();
//     let mut xforms = vec![];
//     let mut sig_len = x.len();
//     for level in 0..j {
//         let w = WaveletXForm1D::<f64>::new(sig_len, lo_d.len());
//         levels[j - level] = w.coeff_len();
//         sig_len = w.coeff_len();
//         xforms.push(w);
//     }
//     *levels.first_mut().unwrap() = xforms.last().unwrap().coeff_len();

//     let mut decomp_len = xforms.iter().fold(0,|acc,x| acc + x.coeff_len());
//     decomp_len += xforms.last().unwrap().coeff_len();

//     println!("decomp_len = {}",decomp_len);
//     //let mut signal = x.to_owned();
//     let mut stop = decomp_len;
//     let mut decomp = vec![Complex64::zero();decomp_len];

//     let mut signal = decomp.clone();
//     decomp[0..x.len()].copy_from_slice(&x);
//     let mut rl = 0;
//     let mut ru = x.len();
//     for xform in xforms.iter_mut() {
//         let start = stop - xform.decomp_len();
//         signal[rl..ru].copy_from_slice(&decomp[rl..ru]);
//         xform.decompose(&signal[rl..ru], &lo_d, &hi_d, &mut decomp[start..stop]);
//         rl = start;
//         ru = start + xform.coeff_len();
//         stop -= xform.coeff_len();
//     }

//     println!("decomp = {:#?}",decomp);

//     // initialization
//     // inputs are decomp and levels
//     let mut start = levels[1];
//     let mut signal = vec![Complex64::zero();x.len()];
//     let mut approx = vec![Complex64::zero();x.len()];
//     let mut xforms:Vec<_> = levels[1..].windows(2).map(|x| {
//         (
//             x[0], // number of approx coeffs
//             x[1], // signal length to reconstruct
//             WaveletXForm1D::new(x[1],4) // wavelet transform handler
//         )
//     }).collect();

//     // the following can be repeated with different decomp arrays without new allocations
//     approx[0..levels[0]].copy_from_slice(&decomp[0..levels[0]]);

//     for(n_coeffs,sig_len,w) in xforms.iter_mut() {
//         let detail = &decomp[start..(start + *n_coeffs)];
//         start += *n_coeffs;
//         w.reconstruct(&approx[0..*n_coeffs], detail, &lo_r, &hi_r, &mut signal[0..*sig_len]);
//         approx[0..*sig_len].copy_from_slice(&signal[0..*sig_len]);
//         //approx = signal[0..sig_len].to_owned();
//         println!("signal = {:#?}",signal);
//     }
// }
