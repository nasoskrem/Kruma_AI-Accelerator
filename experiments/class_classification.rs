use kruma::tensor::Tensor;
use kruma::nn::{Linear, Tanh, Sequential, Module, Dropout};
use kruma::optim::Adam;
use kruma::loss::CrossEntropyLoss;
use kruma::tensor::ops::TensorOps;

fn main() {
    println!("KRUMA: Multi-Class Classification Challenge");
    println!("----------------------------------------------");

    // Batch of 6 samples, 3 features
    let x_train = Tensor::from_data([6, 3], vec![
        1.0, 0.1, 0.0, // Class 0ish
        0.9, 0.0, 0.2, // Class 0ish
        0.1, 1.0, 0.1, // Class 1ish
        0.0, 0.9, 0.2, // Class 1ish
        0.1, 0.1, 1.0, // Class 2ish
        0.2, 0.1, 0.9, // Class 2ish
    ]);
    
    // Targets (One-Hot)
    let y_train = Tensor::from_data([6, 3], vec![
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0, // Class 0
        0.0, 1.0, 0.0, 0.0, 1.0, 0.0, // Class 1
        0.0, 0.0, 1.0, 0.0, 0.0, 1.0, // Class 2
    ]);

    let mut model = Sequential::new(vec![
        Box::new(Linear::new(3, 8)),
        Box::new(Tanh::new()),
        Box::new(Dropout::new(0.2)),
        Box::new(Linear::new(8, 3)),
    ]);

    let mut optimizer = Adam::new(0.01);
    let criterion = CrossEntropyLoss;

    println!("Training for 500 Epochs...");
    for epoch in 1..=10 {
        let logits = model.forward(&x_train);
        let loss = criterion.forward(&logits, &y_train);

        optimizer.zero_grad(&mut model);
        let grad = criterion.backward(&logits, &y_train);
        model.backward(&grad);
        optimizer.step(&mut model);

        if epoch % 100 == 0 { println!("Epoch [{}/500] Loss: {:.6}", epoch, loss); }
    }

    println!("\n✅ Final Evaluation:");
    let logits = model.forward(&x_train);
    let preds = logits.argmax();
    let targets = y_train.argmax();

    for i in 0..6 {
        println!("Sample {}: Pred Class {} | True Class {}", i, preds[i], targets[i]);
    }
}