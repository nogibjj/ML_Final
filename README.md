# Dish Detective 🕵️🥯🥬

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

## Summary

Our study focuses on simplifying dietary tracking using image recognition technology. Three datasets, Food 101/Ingredients 101, Recipes5k, and Nutrition5k, were utilized, with a focus on 11 food types due to computational limitations. A three-model sequence was employed, starting with an InceptionV3 model pretrained on ImageNet for food type prediction, achieving 78% training and 90% validation accuracy. Subsequently, Recipes5k was used to identify ingredients, followed by connecting food types to nutritional values using generative AI like ChatGPT. The model's predictions were compared to Nutrition5k, revealing overestimations in fat, carbs, and protein percentages. To enhance accuracy, the study suggests utilizing comprehensive databases like "FoodData" from the U.S. Department of Agriculture. The final model was integrated into an application for user-friendly food label, ingredient, and nutritional information retrieval from uploaded images.

# Data sources

[Ingredients 101 & Food 101](http://www.ub.edu/cvub/ingredients101/)
[Recipes 5k](http://www.ub.edu/cvub/recipes5k/)

These datasets comes with dish labels and simplified ingredients as well its nutrition percentage by gram.

![eggs](mk_img/food101.png)

[Nutrition 5k](https://console.cloud.google.com/storage/browser/nutrition5k_dataset/nutrition5k_dataset/metadata?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false)

![5k](mk_img/nutrition5k.png)

To get the images from ```dishes_test.txt```  make ```download_images.sh``` executable using ```chmod +x download_images.sh .``` Then, you can run the script using ```./download_images.sh .```

# Results

We ended up building an app from the three models we fine tuned predicting the dish label, the ingredients and the nutrition distribution:

![gif](mk_img/app.gif)


