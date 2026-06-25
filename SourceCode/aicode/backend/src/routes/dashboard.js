const router = require('express').Router();
const ctrl   = require('../controllers/dashboardController');
const { authMiddleware } = require('../middleware/auth');

router.get('/stats',        authMiddleware, ctrl.getStats);
router.get('/weekly',       authMiddleware, ctrl.getWeekly);
router.get('/monthly',      authMiddleware, ctrl.getMonthly);
router.get('/distribution', authMiddleware, ctrl.getDistribution);

module.exports = router;
