from ultralytics import YOLO

model = YOLO("best.pt")

# In ra chính xác những gì model lưu bên trong
print("=== CLASS INDEX TRONG MODEL ===")
for idx in sorted(model.names.keys()):
    print(f"  {idx} → '{model.names[idx]}'  (len={len(model.names[idx])})")

print("\n=== SO SÁNH VỚI CLASS_NAMES TRONG SCRIPT ===")
CLASS_NAMES = ['apple', 'avocado', 'banana', 'dragon fruit',
               'lemon', 'mango', 'orange', 'papaya',
               'pineapple', 'strawberry']

for i, (m, s) in enumerate(zip(list(model.names.values()), CLASS_NAMES)):
    status = "✓" if m == s else f"✗  LỆCH: model='{m}' (len={len(m)})  script='{s}' (len={len(s)})"
    print(f"  Class {i}: {status}")