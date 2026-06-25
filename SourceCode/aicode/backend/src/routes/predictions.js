const router = require('express').Router();
const multer = require('multer');
const ctrl   = require('../controllers/predictionController');
const { authMiddleware } = require('../middleware/auth');

// Keep image in memory (no disk write until after AI processing)
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: parseInt(process.env.MAX_FILE_SIZE || '10485760') },
  fileFilter: (req, file, cb) => {
    const allowed = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
    if (allowed.includes(file.mimetype)) cb(null, true);
    else cb(new Error('Only JPEG, PNG, WebP, BMP images are allowed.'));
  },
});

router.post('/upload',  authMiddleware, upload.single('image'), ctrl.uploadPredict);
router.post('/webcam',  authMiddleware, ctrl.webcamPredict);

module.exports = router;
