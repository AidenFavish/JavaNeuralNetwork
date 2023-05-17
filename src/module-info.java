/**
 * The Java Neural Network is a project designed to help use complex machine learning processes that are often written in Python, in Java. This project helps explore the deep parts of neural networks to make very hands on changes. Efficiency is however not priority number 1 (because if it was you wouldn't be using Java), however multithreading and complex fast matrix operations are coming soon; more compatability and learning features are prioritized higher.
 */
module JavaNeuralNetwork {
    requires json.simple;

    exports com.aidenfavish.javaNeuralNetwork.ActivationFunctions;
    exports com.aidenfavish.javaNeuralNetwork.Layers;
    exports com.aidenfavish.javaNeuralNetwork.Loss;
    exports com.aidenfavish.javaNeuralNetwork.Models;
    exports com.aidenfavish.javaNeuralNetwork.Optimizers;
    exports com.aidenfavish.javaNeuralNetwork.Resources;
}