const router = require('express').Router();
const ctrl   = require('../controllers/adminController');
const { authMiddleware, adminMiddleware } = require('../middleware/auth');

const guard = [authMiddleware, adminMiddleware];

router.get('/users',        ...guard, ctrl.getAllUsers);
router.delete('/users/:id', ...guard, ctrl.deleteUser);
router.get('/records',      ...guard, ctrl.getAllRecords);
router.delete('/records/:id',...guard, ctrl.deleteRecord);
router.get('/stats',        ...guard, ctrl.getSystemStats);

module.exports = router;
