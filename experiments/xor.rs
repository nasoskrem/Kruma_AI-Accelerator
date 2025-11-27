use kruma::tensor::Tensor;
use kruma::nn::{Linear, Tanh, Sequential, Module};
use kruma::optim::Adam;
use kruma::loss::MSELoss;

fn main() {
    println!("KRUMA: XOR Challenge");
    println!("----------------------------------------------");

    // 1. XOR DATASET
    // Inputs: (0,0), (0,1), (1,0), (1,1)
    let x_train = Tensor::from_data([4, 2], vec![
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0
    ]);
    // Targets: 0, 1, 1, 0
    let y_train = Tensor::from_data([4, 1], vec![
        0.0, 
        1.0, 
        1.0, 
        0.0
    ]);

    // 2. MODEL ARCHITECTURE (Sequential)
    // Linear(2 -> 4) -> Tanh -> Linear(4 -> 1) -> Tanh
    let mut model = Sequential::new(vec![
        Box::new(Linear::new(2, 4)), // Hidden Layer
        Box::new(Tanh::new()),       // Non-Linearity
        Box::new(Linear::new(4, 1)), // Output Layer
        Box::new(Tanh::new()),       // Squash to -1..1 (XOR target is 0/1 so this is fine)
    ]);

    // 3. ADVANCED OPTIMIZER (Adam)
    let mut optimizer = Adam::new(0.02); // Fast Learning Rate
    let criterion = MSELoss;

    println!("Training for 500 Epochs...");
    for epoch in 1..=500 {
        // A. Forward
        let pred = model.forward(&x_train);
        let loss = criterion.forward(&pred, &y_train);

        // B. Backward
        optimizer.zero_grad(&mut model);
        let grad = criterion.backward(&pred, &y_train);
        model.backward(&grad);

        // C. Step 
        optimizer.step(&mut model);

        if epoch % 100 == 0 {
            println!("Epoch [{}/500] Loss: {:.6}", epoch, loss);
        }
    }

    println!("\n✅ Final Predictions (Target: 0, 1, 1, 0):");
    let final_pred = model.forward(&x_train);
    for i in 0..4 {
        let val = final_pred[[i, 0]];
        let rounded = if val > 0.5 { 1.0 } else { 0.0 };
        println!("  Input {:?} -> {:.4} (Pred: {})", 
            [x_train[[i,0]], x_train[[i,1]]], val, rounded);
    }
}