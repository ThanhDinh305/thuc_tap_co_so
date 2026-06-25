const path = require('path');
const fs   = require('fs');
const db   = require('../config/db');

// GET /api/history
exports.getHistory = async (req, res) => {
  try {
    const {
      page    = 1,
      limit   = 10,
      search  = '',
      startDate,
      endDate,
    } = req.query;

    const offset = (parseInt(page) - 1) * parseInt(limit);
    const params = [req.user.id];
    let whereClause = 'WHERE user_id = ?';

    if (search) {
      whereClause += ' AND fruit_name LIKE ?';
      params.push(`%${search}%`);
    }
    if (startDate) {
      whereClause += ' AND DATE(created_at) >= ?';
      params.push(startDate);
    }
    if (endDate) {
      whereClause += ' AND DATE(created_at) <= ?';
      params.push(endDate);
    }

    const [countResult] = await db.query(
      `SELECT COUNT(*) as total FROM predictions ${whereClause}`, params
    );
    const total = countResult[0].total;

    const [rows] = await db.query(
      `SELECT * FROM predictions ${whereClause}
       ORDER BY created_at DESC LIMIT ? OFFSET ?`,
      [...params, parseInt(limit), offset]
    );

    res.json({
      success: true,
      data: rows,
      pagination: {
        total,
        page:       parseInt(page),
        limit:      parseInt(limit),
        totalPages: Math.ceil(total / parseInt(limit)),
      },
    });
  } catch (err) {
    console.error('[History] getHistory error:', err);
    res.status(500).json({ success: false, message: 'Server error.' });
  }
};

// GET /api/history/:id
exports.getOne = async (req, res) => {
  try {
    const [rows] = await db.query(
      'SELECT * FROM predictions WHERE id = ? AND user_id = ?',
      [req.params.id, req.user.id]
    );
    if (rows.length === 0)
      return res.status(404).json({ success: false, message: 'Record not found.' });
    res.json({ success: true, data: rows[0] });
  } catch (err) {
    res.status(500).json({ success: false, message: 'Server error.' });
  }
};

// DELETE /api/history/:id
exports.deleteOne = async (req, res) => {
  try {
    const [rows] = await db.query(
      'SELECT * FROM predictions WHERE id = ? AND user_id = ?',
      [req.params.id, req.user.id]
    );
    if (rows.length === 0)
      return res.status(404).json({ success: false, message: 'Record not found.' });

    // Remove image file
    const imgPath = path.join(__dirname, '../../', rows[0].image_path);
    if (fs.existsSync(imgPath)) fs.unlinkSync(imgPath);

    await db.query('DELETE FROM predictions WHERE id = ? AND user_id = ?', [req.params.id, req.user.id]);
    res.json({ success: true, message: 'Deleted successfully.' });
  } catch (err) {
    res.status(500).json({ success: false, message: 'Server error.' });
  }
};

// DELETE /api/history  (delete all for user)
exports.deleteAll = async (req, res) => {
  try {
    const [rows] = await db.query('SELECT image_path FROM predictions WHERE user_id = ?', [req.user.id]);
    // Remove image files
    rows.forEach(row => {
      const imgPath = path.join(__dirname, '../../', row.image_path);
      if (fs.existsSync(imgPath)) fs.unlinkSync(imgPath);
    });

    await db.query('DELETE FROM predictions WHERE user_id = ?', [req.user.id]);
    res.json({ success: true, message: `Deleted ${rows.length} records.` });
  } catch (err) {
    res.status(500).json({ success: false, message: 'Server error.' });
  }
};

// GET /api/history/export/excel
exports.exportExcel = async (req, res) => {
  try {
    const ExcelJS = require('exceljs');
    const [rows] = await db.query(
      'SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC',
      [req.user.id]
    );

    const workbook  = new ExcelJS.Workbook();
    const worksheet = workbook.addWorksheet('Fruit History');

    worksheet.columns = [
      { header: 'ID',            key: 'id',            width: 8  },
      { header: 'Fruit',         key: 'fruit_name',    width: 15 },
      { header: 'Confidence (%)',key: 'confidence',    width: 16 },
      { header: 'Ripeness',      key: 'ripeness',      width: 15 },
      { header: 'Calories',      key: 'calories',      width: 12 },
      { header: 'Protein (g)',   key: 'protein',       width: 13 },
      { header: 'Fat (g)',       key: 'fat',           width: 10 },
      { header: 'Carbs (g)',     key: 'carbohydrates', width: 13 },
      { header: 'Date',          key: 'created_at',    width: 22 },
    ];

    // Style header row
    worksheet.getRow(1).eachCell(cell => {
      cell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FF10b981' } };
      cell.font = { color: { argb: 'FFFFFFFF' }, bold: true };
    });

    rows.forEach(row => worksheet.addRow(row));

    res.setHeader('Content-Type', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');
    res.setHeader('Content-Disposition', 'attachment; filename=fruit_history.xlsx');
    await workbook.xlsx.write(res);
    res.end();
  } catch (err) {
    console.error('[History] exportExcel error:', err);
    res.status(500).json({ success: false, message: 'Export failed.' });
  }
};

// GET /api/history/export/pdf
exports.exportPDF = async (req, res) => {
  try {
    const { jsPDF } = require('jspdf');
    require('jspdf-autotable');

    const [rows] = await db.query(
      'SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC',
      [req.user.id]
    );

    const doc  = new jsPDF({ orientation: 'landscape' });
    doc.setFontSize(16);
    doc.text('Fruit Recognition History', 14, 15);
    doc.setFontSize(10);
    doc.text(`Generated: ${new Date().toLocaleString()}`, 14, 22);

    doc.autoTable({
      startY: 28,
      head: [['Fruit', 'Confidence', 'Ripeness', 'Calories', 'Protein', 'Fat', 'Carbs', 'Date']],
      body: rows.map(r => [
        r.fruit_name,
        `${r.confidence}%`,
        r.ripeness || '-',
        r.calories ? `${r.calories} kcal` : '-',
        r.protein  ? `${r.protein} g`     : '-',
        r.fat      ? `${r.fat} g`         : '-',
        r.carbohydrates ? `${r.carbohydrates} g` : '-',
        new Date(r.created_at).toLocaleDateString(),
      ]),
      styles: { fontSize: 8 },
      headStyles: { fillColor: [16, 185, 129] },
    });

    const buffer = Buffer.from(doc.output('arraybuffer'));
    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', 'attachment; filename=fruit_history.pdf');
    res.send(buffer);
  } catch (err) {
    console.error('[History] exportPDF error:', err);
    res.status(500).json({ success: false, message: 'PDF export failed.' });
  }
};
