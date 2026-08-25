V8.5.26 model roles

best.pt          : vehicle detector (creates the vehicle bounding box)
tire_level.pt    : tire flood-stage detector, run first on each vehicle crop
car_flood_cls.pt : vehicle-body flood-stage classifier, used ONLY when tire_level detects no tire

Pipeline:
frame -> best.pt vehicle boxes -> crop each vehicle -> tire_level.pt
      -> tire detected: use tire stage
      -> no tire detected: car_flood_cls.pt body fallback
      -> keep the original best.pt vehicle bbox for screen drawing

The tire_level.pt in this package is the model uploaded by the user for V8.5.26.
