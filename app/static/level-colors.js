/*
 * POHANG flood level palette V8.3.7
 *
 * This is the single color source used by:
 * - DEM flood depth polygons
 * - VWorld flood depth polygons
 * - CCTV map points
 * - CCTV list levels
 * - CCTV video level badges/charts
 * - recent flood warning cards
 * - event focus markers
 */
window.POHANG_LEVEL_COLORS = Object.freeze([
  '#42c889', // Lev0 정상
  '#315cff', // Lev1 1~11 cm
  '#7136d9', // Lev2 12~34 cm
  '#bd2caf', // Lev3 35~59 cm
  '#ff4298'  // Lev4 60 cm 이상
]);
