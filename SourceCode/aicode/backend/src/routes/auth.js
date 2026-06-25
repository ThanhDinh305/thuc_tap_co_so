const router  = require('express').Router();
const ctrl    = require('../controllers/authController');
const { authMiddleware } = require('../middleware/auth');

router.post('/register',    ctrl.register);
router.post('/login',       ctrl.login);
router.get('/me',           authMiddleware, ctrl.getMe);
router.put('/profile',      authMiddleware, ctrl.updateProfile);

module.exports = router;
