# Preprocessing Pipeline (summary)

## Dropped features
(none)

## Encodings used
- OneHotEncoder (low-cardinality categorical)
  - Features : Doors, Drive wheels, Gear box type, Category, Color, Fuel type
- OneHotEncoder (binary categorical)
  - Features : Leather interior, Wheel
- TargetEncoder (high-cardinality categorical)
  - Features : Manufacturer, Model

## Transformations applied
- Levy
  - Replace missing data indicator (-) with nan
  - Cast to float
  - Impute with median
  - log1p transform
  - Robust scaling
- Mileage
  - Remove " km" from values
  - Cast to float
  - Impute with median
  - log1p transform
  - Robust scaling
- Engine volume
  - Split to Engine volume (float) and Turbo (int) features
  - Engine volume: impute with median, standardize
  - Turbo: impute with most frequent, cast to int

## Known limitations
- Extreme outliers in Levy and Engine volume are preserved (no clipping)
