<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/AidenFavish/JavaNeuralNetwork">
    <img src="README_images/logo.png" alt="Logo" width="256" height="256">
  </a>

<h3 align="center">Java Neural Network</h3>

  <p align="center">
    A completely vanilla Java Neural Network inspired by nnfs. Optimized for categorical inference. WORK IN PROGRESS.
    <br />
    <a href="https://aidenfavish.github.io/JavaNeuralNetwork"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/AidenFavish/JavaNeuralNetwork">View Demo</a>
    ·
    <a href="https://github.com/AidenFavish/JavaNeuralNetwork/issues">Report Bug</a>
    ·
    <a href="https://github.com/AidenFavish/JavaNeuralNetwork/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

```java
import com.aidenfavish.javaNeuralNetwork.ActivationFunctions.ActivationELU;
import com.aidenfavish.javaNeuralNetwork.ActivationFunctions.ActivationReLU;
import com.aidenfavish.javaNeuralNetwork.Layers.LayerDense;
import com.aidenfavish.javaNeuralNetwork.Loss.ActivationSoftMaxCCE;
import com.aidenfavish.javaNeuralNetwork.Models.Model;
import com.aidenfavish.javaNeuralNetwork.Optimizers.AdamOptimizer;

/** All it takes to train and test a model
 *
 */
public class Tester {
    public static void main(String[] args) {
        Model model = new Model("TesterModel", new AdamOptimizer(0.01f, (float) (5 * Math.pow(10, -5)), (float) (1 * Math.pow(10, -7)), 0.9f, 0.999f));
        model.addLayer(new LayerDense(2, 8));
        model.addLayer(new ActivationReLU());
        model.addLayer(new LayerDense(8, 4));
        model.addLayer(new ActivationELU());
        model.addLayer(new LayerDense(4, 3));
        model.addLayer(new ActivationSoftMaxCCE());

        model.save("Models/TesterModel.json");

        model = new Model("Models/TesterModel.json");
        model.train(500, new float[][]{{0.1f, 0.1f}, {0.1f, 0.1f}}, new int[]{1, 1});
        System.out.println(model.predict(new float[]{0.1f, 0.11f}));
    }
}
```

The Java Neural Network is a project designed to help use complex machine learning processes that are often written in Python, in Java. This project helps explore the deep parts of neural networks to make very hands on changes. Efficiency is however not priority number 1 (because if it was you wouldn't be using Java), however multithreading and complex fast matrix operations are coming soon; more compatability and learning features are prioritized higher. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Installation

1. Download the jar files or the classes individually to get started
2. Import the jar files or classes into your project and start working
3. Getting started with the model class will be the easiest way to start learning the network syntax.
4. [Use our JavaDoc](https://aidenfavish.github.io/JavaNeuralNetwork)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

The Java Neural Network is perfect for training using Reinforcement Learning Techniques on your own custom Java environments. It has also been built to perform supervised learning with a wide variety of different options for optimizers and activation functions.

_For more examples, please refer to the [Documentation](https://aidenfavish.github.io/JavaNeuralNetwork)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [ ] PPO
- [ ] A2C
- [ ] Evolution Learning
- [ ] More Activation Functions
- [ ] Easier inheritance to create custom activation functions

See the [open issues](https://github.com/AidenFavish/JavaNeuralNetwork/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Aiden Favish - Twitter: [@AidenFavish](https://twitter.com/@AidenFavish)

Project Link: [https://github.com/AidenFavish/JavaNeuralNetwork](https://github.com/AidenFavish/JavaNeuralNetwork)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []() NNFS - https://nnfs.io/

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/AidenFavish/JavaNeuralNetwork.svg?style=for-the-badge
[contributors-url]: https://github.com/AidenFavish/JavaNeuralNetwork/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/AidenFavish/JavaNeuralNetwork.svg?style=for-the-badge
[forks-url]: https://github.com/AidenFavish/JavaNeuralNetwork/network/members
[stars-shield]: https://img.shields.io/github/stars/AidenFavish/JavaNeuralNetwork.svg?style=for-the-badge
[stars-url]: https://github.com/AidenFavish/JavaNeuralNetwork/stargazers
[issues-shield]: https://img.shields.io/github/issues/AidenFavish/JavaNeuralNetwork.svg?style=for-the-badge
[issues-url]: https://github.com/AidenFavish/JavaNeuralNetwork/issues
[license-shield]: https://img.shields.io/github/license/AidenFavish/JavaNeuralNetwork.svg?style=for-the-badge
[license-url]: https://github.com/AidenFavish/JavaNeuralNetwork/blob/master/LICENSE.txt
[product-screenshot]: images/screenshot.png
