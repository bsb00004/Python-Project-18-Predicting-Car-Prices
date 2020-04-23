## Python Project 18: Predicting Car Prices
In this project, we'll use the machine learning workflow (K-nearest neighbour) to predict a car's market price using its attributes. The data set we will be working with contains information on various cars. For each car we have information about the technical aspects of the vehicle such as the motor's displacement, the weight of the car, the miles per gallon, how fast the car accelerates, and more. You can read more about the data set [here](https://archive.ics.uci.edu/ml/datasets/automobile). Here's a documentation of the data set:

| Attribute         | Attribute Range                                                                                                                                                                                |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| symboling         | -3, -2, -1, 0, 1, 2, 3.                                                                                                                                                                        |
| normalized-losses | continuous from 65 to 256.                                                                                                                                                                     |
| make              | alfa-romero, audi, bmw, chevrolet, dodge, honda, isuzu, jaguar, mazda, mercedes-benz, mercury, mitsubishi, nissan, peugot, plymouth, porsche, renault, saab, subaru, toyota, volkswagen, volvo |
| fuel-type         | diesel, gas.                                                                                                                                                                                   |
| aspiration        | std, turbo.                                                                                                                                                                                    |
| num-of-doors      | four, two.                                                                                                                                                                                     |
| body-style        | hardtop, wagon, sedan, hatchback, convertible.                                                                                                                                                 |
| drive-wheels      | 4wd, fwd, rwd.                                                                                                                                                                                 |
| engine-location   | front, rear.                                                                                                                                                                                   |
| wheel-base        | continuous from 86.6 120.9.                                                                                                                                                                    |
| length            | continuous from 141.1 to 208.1.                                                                                                                                                                |
| width             | continuous from 60.3 to 72.3.                                                                                                                                                                  |
| height            | continuous from 47.8 to 59.8.                                                                                                                                                                  |
| curb-weight       | continuous from 1488 to 4066.                                                                                                                                                                  |
| engine-type       | dohc, dohcv, l, ohc, ohcf, ohcv, rotor.                                                                                                                                                        |
| num-of-cylinders  | eight, five, four, six, three, twelve, two.                                                                                                                                                    |
| engine-size       | continuous from 61 to 326.                                                                                                                                                                     |
| fuel-system       | 1bbl, 2bbl, 4bbl, idi, mfi, mpfi, spdi, spfi.                                                                                                                                                  |
| bore              | continuous from 2.54 to 3.94.                                                                                                                                                                  |
| stroke            | continuous from 2.07 to 4.17.                                                                                                                                                                  |
| compression-ratio | continuous from 7 to 23.                                                                                                                                                                       |
| horsepower        | continuous from 48 to 288.                                                                                                                                                                     
| peak-rpm          | continuous from 4150 to 6600.                                                                                                                                                                  |
| city-mpg          | continuous from 13 to 49.                                                                                                                                                                      |
| highway-mpg       | continuous from 16 to 54.                                                                                                                                                                      |
| price             | continuous from 5118 to 45400.                                                                                                                                                                 |

      
- `price` column is the one we want to predict.

## Note
### - Please see the `K-nearest.ipynb` file to see whole project in detail.
### - Please see `K-nearest.py` file to see the python code.
### - `imports-85.data` is the dataset we used in this project.
