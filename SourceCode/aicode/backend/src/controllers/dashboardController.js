const db = require('../config/db');

// GET /api/dashboard/stats
exports.getStats = async (req, res) => {
  try {
    const uid = req.user.id;
    const [[{ total }]]   = await db.query('SELECT COUNT(*) as total FROM predictions WHERE user_id = ?', [uid]);
    const [[{ today }]]   = await db.query("SELECT COUNT(*) as today FROM predictions WHERE user_id = ? AND DATE(created_at) = CURDATE()", [uid]);
    const [[{ thisWeek }]]= await db.query("SELECT COUNT(*) as thisWeek FROM predictions WHERE user_id = ? AND YEARWEEK(created_at, 1) = YEARWEEK(CURDATE(), 1)", [uid]);
    const [topFruitRows]  = await db.query(
      "SELECT fruit_name, COUNT(*) as count FROM predictions WHERE user_id = ? GROUP BY fruit_name ORDER BY count DESC LIMIT 1",
      [uid]
    );
    const topFruit = topFruitRows.length > 0 ? topFruitRows[0] : null;

    res.json({ success: true, data: { total, today, thisWeek, topFruit } });
  } catch (err) {
    console.error('[Dashboard] getStats error:', err);
    res.status(500).json({ success: false, message: 'Server error.' });
  }
};

// GET /api/dashboard/weekly
exports.getWeekly = async (req, res) => {
  try {
    const [rows] = await db.query(
      `SELECT DATE(created_at) as date, COUNT(*) as count
       FROM predictions
       WHERE user_id = ? AND created_at >= DATE_SUB(CURDATE(), INTERVAL 6 DAY)
       GROUP BY DATE(created_at)
       ORDER BY date ASC`,
      [req.user.id]
    );

    // Fill all 7 days with 0 if missing
    const result = [];
    for (let i = 6; i >= 0; i--) {
      const d = new Date();
      d.setDate(d.getDate() - i);
      const dateStr = d.toISOString().slice(0, 10);
      const found = rows.find(r => r.date.toISOString().slice(0, 10) === dateStr);
      result.push({ date: dateStr, count: found ? found.count : 0 });
    }

    res.json({ success: true, data: result });
  } catch (err) {
    res.status(500).json({ success: false, message: 'Server error.' });
  }
};

// GET /api/dashboard/monthly
exports.getMonthly = async (req, res) => {
  try {
    const [rows] = await db.query(
      `SELECT DATE_FORMAT(created_at, '%Y-%m') as month, COUNT(*) as count
       FROM predictions
       WHERE user_id = ? AND created_at >= DATE_SUB(CURDATE(), INTERVAL 11 MONTH)
       GROUP BY month
       ORDER BY month ASC`,
      [req.user.id]
    );

    // Fill 12 months
    const result = [];
    for (let i = 11; i >= 0; i--) {
      const d = new Date();
      d.setMonth(d.getMonth() - i);
      const monthStr = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`;
      const found = rows.find(r => r.month === monthStr);
      result.push({ month: monthStr, count: found ? found.count : 0 });
    }

    res.json({ success: true, data: result });
  } catch (err) {
    res.status(500).json({ success: false, message: 'Server error.' });
  }
};

// GET /api/dashboard/distribution
exports.getDistribution = async (req, res) => {
  try {
    const [rows] = await db.query(
      `SELECT fruit_name, COUNT(*) as count
       FROM predictions WHERE user_id = ?
       GROUP BY fruit_name ORDER BY count DESC`,
      [req.user.id]
    );
    res.json({ success: true, data: rows });
  } catch (err) {
    res.status(500).json({ success: false, message: 'Server error.' });
  }
};
