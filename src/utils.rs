pub fn set_panic_hook() {
    // When the `console_error_panic_hook` feature is enabled, we can call the
    // `set_panic_hook` function at least once during initialization, and then
    // we will get better error messages if our code ever panics.
    //
    // For more details see
    // https://github.com/rustwasm/console_error_panic_hook#readme
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

pub fn unique_vals(arr: &Vec<String>) -> Vec<String> {
    let mut u_vals: Vec<String> = vec![];
    for el in arr.iter() {
        if !u_vals.contains(&el) {
            u_vals.push(el.to_string());
        }
    }
    u_vals
}

pub fn unique_vals_f32(arr: &Vec<f32>) -> Vec<f32> {
    let mut u_vals: Vec<f32> = vec![];
    for el in arr.iter() {
        if !u_vals.contains(&el) {
            u_vals.push(*el);
        }
    }
    u_vals
}

pub fn count_vals(arr: &Vec<String>, label: String) -> usize {
    let mut c = 0;
    for el in arr.iter() {
        if el == &label {
            c = c + 1;
        }
    }
    
    c
}

pub fn get_column(matrix: &Vec<Vec<f32>>, col: usize) -> Vec<f32>{
    let mut column: Vec<f32> = vec![];
    for row in matrix.iter() {
        for (j, &colu) in row.iter().enumerate() {
            if j == col {
                //println!("{} {}", j, col);
                column.push(colu);
            }
        }
    }
    column
}