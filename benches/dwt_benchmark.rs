use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dwt::{
    w_max_level,
    wavelet::{Wavelet, WaveletFilter, WaveletType},
    WaveDecPlanner, WaveRecPlanner,
};
use num_complex::Complex64;
use num_traits::{One, Zero};

pub fn criterion_benchmark(c: &mut Criterion) {
    let n = 788;
    let x = vec![Complex64::one(); n];
    let wavelet = Wavelet::<f64>::new(WaveletType::Daubechies2);
    let n_levels = w_max_level(x.len(), wavelet.filt_len());
    println!("max decomp levels: {}", n_levels);
    let mut wavedec = WaveDecPlanner::<f64>::new(n, n_levels, wavelet);
    let mut waverec = WaveRecPlanner::<f64>::new(&wavedec);

    let mut result = vec![Complex64::zero(); wavedec.decomp_len()];
    let mut recon = vec![Complex64::zero(); x.len()];

    c.bench_function("wavlet decomp", |b| {
        b.iter(|| {
            wavedec.process(black_box(&x), black_box(&mut result));
        })
    });

    c.bench_function("wavlet recon", |b| {
        b.iter(|| {
            waverec.process(black_box(&result), black_box(&mut recon));
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
