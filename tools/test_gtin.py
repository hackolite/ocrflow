import gtin
from gtin import has_valid_check_digit
from gtin import has_valid_check_digit
gtin_number = "3036350259606"

if has_valid_check_digit(gtin_number):
    print(f"Le GTIN {gtin_number} est valide.")
else:
    print(f"Le GTIN {gtin_number} n'est pas valide.")
