# Preprocessing Pipeline (summary)

## Dropped features
- Model
  - Reason : High cardinality (n = 1180)

## Encodings used
- OneHotEncoder (low-cardinality categorical)
  - Features : Doors, Drive wheels, Gear box type, Category, Color, Fuel type
- OneHotEncoder (binary categorical)
  - Features : Leather interior, Wheel

## Transformations applied
- Levy
  - Replace missing data indicator (-) with nan
  - Cast to float
  - Impute with median
  - Standardize
- Mileage
  - Remove " km" from values
  - Cast to float
  - Impute with median
  - log1p transform
  - Standardize
- Engine volume
  - Split to Engine volume (float) and Turbo (int) features
  - Engine volume: impute with median, standardize
  - Turbo: impute with most frequent, cast to int
- Manufacturer
  - Impute with most frequent
  - Map to country of origin
  - One-hot encode

## Known limitations
- Brand significance is no longer a learning signal
- Multinational branding not encoded
- Extreme outliers in Levy and Engine volume are preserved (no clipping)
