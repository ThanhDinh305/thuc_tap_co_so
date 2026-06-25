const router = require('express').Router();
const ctrl   = require('../controllers/historyController');
const { authMiddleware } = require('../middleware/auth');

router.get('/export/excel', authMiddleware, ctrl.exportExcel);
router.get('/export/pdf',   authMiddleware, ctrl.exportPDF);
router.get('/',             authMiddleware, ctrl.getHistory);
router.get('/:id',          authMiddleware, ctrl.getOne);
router.delete('/',          authMiddleware, ctrl.deleteAll);
router.delete('/:id',       authMiddleware, ctrl.deleteOne);

module.exports = router;
