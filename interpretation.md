# Interpretations


Used learning rate of 0.01, which is standard, for updating weight and biases in Perceptron code.

All features in the dataset were used, other than Magnesium and Proline levels due to the difference in scale. In the future, these features could be useful to derive a more accurate model, assuming a scaler is applied. 

* The model precison for classes 1,2, and 3 are: 91% 81% and 100% respectively.
* The model recall for classes 1,2, and 3 are: 83% 93% and 90% respectively.

This indicates that the model was best at calculating true-positives within class 3; it was best at calculating true-positives in class 2 when considering only class similar observations. 

Overall this model preformed well usin the multi-class perceptron, but may benefit from other learning algorithms like KNN due to its non-binary nature.

