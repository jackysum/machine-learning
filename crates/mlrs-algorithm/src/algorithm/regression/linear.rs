pub struct Linear {
    slope: Option<f64>,
    intercept: Option<f64>,
}

impl Linear {
    pub fn new() -> Self {
        Linear {
            slope: None,
            intercept: None,
        }
    }

    pub fn fit(&mut self, x: &[f64], y: &[f64]) {
        assert!(
            self.intercept.is_none() && self.slope.is_none(),
            "model is already fitted"
        );
        assert_eq!(x.len(), y.len(), "input vectors must have the same length");
        assert!(x.len() > 1, "input vectors must have more than one element");

        let n = x.len() as f64;
        let mut sum_x: f64 = 0.0;
        let mut sum_y: f64 = 0.0;
        let mut sum_x_squared: f64 = 0.0;
        let mut sum_xy: f64 = 0.0;

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            sum_x += xi;
            sum_y += yi;
            sum_x_squared += xi * xi;
            sum_xy += xi * yi;
        }

        self.intercept =
            Some((sum_y * sum_x_squared - sum_x * sum_xy) / (n * sum_x_squared - sum_x * sum_x));

        self.slope = Some((n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x * sum_x));
    }

    pub fn predict(&self, values: &[f64]) -> Vec<f64> {
        assert!(
            self.intercept.is_some() && self.slope.is_some(),
            "model must be fitted before prediction"
        );

        values
            .iter()
            .map(|&x| self.intercept.unwrap() + self.slope.unwrap() * x)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use crate::LinearRegression;

    #[test]
    #[should_panic(expected = "model is already fitted")]
    fn test_linear_fit_model_is_already_fitted() {
        let mut model = LinearRegression::new();
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![5.0, 6.0, 7.0];

        model.fit(&x, &y);
        model.fit(&x, &y);
    }

    #[test]
    #[should_panic(expected = "input vectors must have the same length")]
    fn test_linear_fit_input_vectors_different_length() {
        let mut model = LinearRegression::new();
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![5.0, 6.0];

        model.fit(&x, &y);
    }

    #[test]
    #[should_panic(expected = "input vectors must have more than one element")]
    fn test_linear_fit_input_vectors_too_short() {
        let mut model = LinearRegression::new();
        let x = vec![1.0];
        let y = vec![5.0];

        model.fit(&x, &y);
    }

    #[test]
    fn test_linear_predict() {
        let mut model = LinearRegression::new();
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![5.0, 6.0, 7.0];

        model.fit(&x, &y);

        let want = 8.0;
        let got = model.predict(&[4.0])[0];

        assert_eq!(got, want, "wanted {}, got {}", want, got);
    }

    #[test]
    fn test_linear_predict_lifesat() {
        let file = File::open("Datasets/lifesat.csv").expect("error opening csv file");
        let mut rdr = csv::Reader::from_reader(file);

        let mut x_values = Vec::new();
        let mut y_values = Vec::new();

        for result in rdr.records() {
            let record = result.expect("error reading record");
            let x: f64 = record
                .get(1)
                .unwrap()
                .parse()
                .expect("error parsing x value");

            let y: f64 = record
                .get(2)
                .unwrap()
                .parse()
                .expect("error parsing y value");

            x_values.push(x);
            y_values.push(y);
        }

        let mut model = LinearRegression::new();
        model.fit(&x_values, &y_values);

        let want = 6.301657665080505;
        let got = model.predict(&[37_655.2])[0];

        assert_eq!(got, want, "wanted {}, got {}", want, got);
    }

    #[test]
    #[should_panic(expected = "model must be fitted before prediction")]
    fn test_linear_predict_model_not_fitted() {
        let model = LinearRegression::new();
        model.predict(&[1.0]);
    }
}
