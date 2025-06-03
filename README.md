# LundTagger
This repository provides a framework for jet tagging at the FCC using a [LundNet](https://arxiv.org/abs/2012.08526)-based Graph Neural Network. The code heavily relies on the [SVJ-GNN](https://github.com/rseidita/SVJ_GNN), with a few changes; nevertheless I warmely suggest to take a look at it before using this repo, particularly for setting up the correct environment. The core of the code is a `LundTagger` object, where one should specify:

- `n`: the amount of classes the tagger discriminates. This is an integer between 4 and 6. For each of these values the classes are:
  - 4: light quarks, s-quark, c-quark, b-quark
  - 5: light quarks, s-quark, c-quark, b-quark, gluons
  - 6: u-quarks, d-quarks, s-quark, c-quark, b-quark, gluons
- `modelname`: the name of the script where the model is specify. The code looks for this is script in the `LundTagger/architectures/` folder and expects a class `LundNetTagger` where the model would be contained. This is set by default to `arch`. An example of architecture is shown in `./architectures/my_arch.py`.
- `epochs`: the amount of epochs you want to train or load the model you are working with. It is set by default to 150
- `pdg`: a bool that specifies whether you want to use the PDG information in your tree
- `suffix`: a string which will be added at the end of any saved file. This is useful, for instance, when loading different `JetGraphProducer`s. 

This repo provides the tools to train, evaluate and compare several models. A few samples with Higgsstrahlung events where a Higgs decays to quaks and the Z to a muon pair are present in the `LundTagger/samples` folder. You can refer to `basic_example.py` to check a few functionalities and how to use them. Please do not hesitate to contact me for questions or issues. 
