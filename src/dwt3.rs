#[test]
fn dwt_down() {
    let x: Vec<_> = (1..=11).map(|x| x as f32).collect();

    let lo_d = [-0.1294, 0.2241, 0.8365, 0.4830];
    let hi_d = [-0.4830, 0.8365, -0.2241, -0.1294];
    let lo_r = [0.4830, 0.8365, 0.2241, -0.1294];
    let hi_r = [-0.1294, -0.2241, 0.8365, -0.4830];

    let f_len = 4;
    let sig_len = x.len() as i32;

    let n_coeffs = (sig_len + f_len - 1) / 2;
    let ext_len = f_len as i32 - 1;

    let mut result = vec![0.; n_coeffs as usize * 2];

    for i in 0..n_coeffs as i32 {
        let mut sa = 0.;
        let mut sd = 0.;
        for j in 0..f_len as i32 {
            // calculate virtual signal index that may extend out-of-bounds
            let virtual_idx = 2 * i - ext_len + j + 1;
            // rationalize the index by imposing the boundary condition
            let signal_idx = symmetric_boundary_index(virtual_idx, sig_len);
            // calculate the filter index
            let filter_idx = (f_len - j - 1) as usize;
            // multiply signal with filter coefficient
            sa += lo_d[filter_idx] * x[signal_idx];
            sd += hi_d[filter_idx] * x[signal_idx];
        }
        result[i as usize] = sa;
        result[n_coeffs as usize + i as usize] = sd;
    }

    println!("result: {:?}", result);
}


fn symmetric_boundary_index(index: i32, n: i32) -> usize {
    if index < 0 {
        (index + 1).abs() as usize
    } else if index >= n {
        (2 * n - index - 1) as usize
    } else {
        index as usize
    }
}

#[test]
fn test_symmetric_boundary() {
    println!("{:?}", symmetric_boundary_index(0, 4));
    println!("{:?}", symmetric_boundary_index(5, 4));
    println!("{:?}", symmetric_boundary_index(-2, 4));
}