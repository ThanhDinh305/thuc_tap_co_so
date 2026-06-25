const db = require('../config/db');
const fs = require('fs');
const path = require('path');

// GET /api/admin/users
exports.getAllUsers = async (req, res) => {
  try {
    const { page = 1, limit = 20, search = '' } = req.query;
    const offset = (parseInt(page) - 1) * parseInt(limit);
    const params = [];
    let where = '';
    if (search) {
      where = 'WHERE name LIKE ? OR email LIKE ?';
      params.push(`%${search}%`, `%${search}%`);
    }

    const [[{ total }]] = await db.query(`SELECT COUNT(*) as total FROM users ${where}`, params);
    const [users] = await db.query(
      `SELECT id, name, email, role, created_at,
         (SELECT COUNT(*) FROM predictions WHERE user_id = users.id) as prediction_count
       FROM users ${where} ORDER BY created_at DESC LIMIT ? OFFSET ?`,
      [...params, parseInt(limit), offset]
    );

    res.json({
      success: true, data: users,
      pagination: { total, page: parseInt(page), limit: parseInt(limit), totalPages: Math.ceil(total / parseInt(limit)) }
    });
  } catch (err) {
    console.error('[Admin] getAllUsers error:', err);
    res.status(500).json({ success: false, message: 'Server error.' });
  }
};

// DELETE /api/admin/users/:id
exports.deleteUser = async (req, res) => {
  try {
    if (parseInt(req.params.id) === req.user.id)
      return res.status(400).json({ success: false, message: 'Cannot delete yourself.' });

    const [rows] = await db.query('SELECT * FROM predictions WHERE user_id = ?', [req.params.id]);
    rows.forEach(row => {
      const imgPath = path.join(__dirname, '../../', row.image_path);
      if (fs.existsSync(imgPath)) fs.unlinkSync(imgPath);
    });

    await db.query('DELETE FROM users WHERE id = ?', [req.params.id]);
    res.json({ success: true, message: 'User deleted.' });
  } catch (err) {
    res.status(500).json({ success: false, message: 'Server error.' });
  }
};

// GET /api/admin/records
exports.getAllRecords = async (req, res) => {
  try {
    const { page = 1, limit = 20, search = '', startDate, endDate } = req.query;
    const offset = (parseInt(page) - 1) * parseInt(limit);
    const params = [];
    let where = '';
    const conditions = [];

    if (search) {
      conditions.push('(p.fruit_name LIKE ? OR u.name LIKE ?)');
      params.push(`%${search}%`, `%${search}%`);
    }
    if (startDate) { conditions.push('DATE(p.created_at) >= ?'); params.push(startDate); }
    if (endDate)   { conditions.push('DATE(p.created_at) <= ?'); params.push(endDate);   }
    if (conditions.length) where = 'WHERE ' + conditions.join(' AND ');

    const [[{ total }]] = await db.query(
      `SELECT COUNT(*) as total FROM predictions p JOIN users u ON p.user_id = u.id ${where}`, params
    );
    const [records] = await db.query(
      `SELECT p.*, u.name as user_name, u.email as user_email
       FROM predictions p JOIN users u ON p.user_id = u.id
       ${where} ORDER BY p.created_at DESC LIMIT ? OFFSET ?`,
      [...params, parseInt(limit), offset]
    );

    res.json({
      success: true, data: records,
      pagination: { total, page: parseInt(page), limit: parseInt(limit), totalPages: Math.ceil(total / parseInt(limit)) }
    });
  } catch (err) {
    res.status(500).json({ success: false, message: 'Server error.' });
  }
};

// DELETE /api/admin/records/:id
exports.deleteRecord = async (req, res) => {
  try {
    const [rows] = await db.query('SELECT * FROM predictions WHERE id = ?', [req.params.id]);
    if (rows.length === 0)
      return res.status(404).json({ success: false, message: 'Record not found.' });

    const imgPath = path.join(__dirname, '../../', rows[0].image_path);
    if (fs.existsSync(imgPath)) fs.unlinkSync(imgPath);

    await db.query('DELETE FROM predictions WHERE id = ?', [req.params.id]);
    res.json({ success: true, message: 'Record deleted.' });
  } catch (err) {
    res.status(500).json({ success: false, message: 'Server error.' });
  }
};

// GET /api/admin/stats
exports.getSystemStats = async (req, res) => {
  try {
    const [[{ totalUsers }]]   = await db.query("SELECT COUNT(*) as totalUsers FROM users WHERE role = 'user'");
    const [[{ totalPredictions }]] = await db.query("SELECT COUNT(*) as totalPredictions FROM predictions");
    const [[{ today }]]        = await db.query("SELECT COUNT(*) as today FROM predictions WHERE DATE(created_at) = CURDATE()");
    const [[{ thisWeek }]]     = await db.query("SELECT COUNT(*) as thisWeek FROM predictions WHERE YEARWEEK(created_at, 1) = YEARWEEK(CURDATE(), 1)");

    const [fruitDist] = await db.query(
      "SELECT fruit_name, COUNT(*) as count FROM predictions GROUP BY fruit_name ORDER BY count DESC"
    );
    const [recentActivity] = await db.query(
      `SELECT p.fruit_name, p.confidence, p.created_at, u.name as user_name
       FROM predictions p JOIN users u ON p.user_id = u.id
       ORDER BY p.created_at DESC LIMIT 10`
    );

    res.json({
      success: true,
      data: { totalUsers, totalPredictions, today, thisWeek, fruitDist, recentActivity }
    });
  } catch (err) {
    res.status(500).json({ success: false, message: 'Server error.' });
  }
};
