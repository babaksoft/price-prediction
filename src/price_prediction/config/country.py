_country_manufacturer_map = {
  "United States": [
      "LINCOLN", "CHEVROLET", "FORD", "DODGE", "CHRYSLER", "JEEP",
      "CADILLAC", "SCION", "HUMMER", "BUICK", "GMC", "SATURN",
      "MERCURY", "TESLA", "PONTIAC"
  ],
  "South Korea": ["KIA", "HYUNDAI", "SSANGYONG", "DAEWOO"],
  "Germany": [
      "VOLKSWAGEN", "MERCEDES-BENZ", "AUDI", "BMW", "OPEL", "PORSCHE"
  ],
  "Japan": [
      "TOYOTA", "ACURA", "HONDA", "LEXUS", "MAZDA", "NISSAN", "SUBARU",
      "MITSUBISHI", "SUZUKI", "INFINITI", "DAIHATSU", "ISUZU"
  ],
  "Russia": ["VAZ", "GAZ", "MOSKVICH", "UAZ"],
  "Italy": [
      "ALFA ROMEO", "MASERATI", "FIAT", "FERRARI", "LAMBORGHINI"
  ],
  "France": ["RENAULT", "CITROEN", "PEUGEOT"],
  "Czech Republic": ["SKODA"],
  "Ukraine": ["ZAZ"],
  "United Kingdom": [
      "MINI", "JAGUAR", "LAND ROVER", "ROVER", "ASTON MARTIN"
  ],
  "Sweden": ["VOLVO", "SAAB"],
  "China": ["GREATWALL"],
  "Spain": ["SEAT"]
}


def get_country_map():
    country_map = {}
    for country in _country_manufacturer_map:
        mapping = { manufacturer: country
                    for manufacturer in _country_manufacturer_map[country] }
        country_map.update(mapping)

    return country_map
