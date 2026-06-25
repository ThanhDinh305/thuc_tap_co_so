const path    = require('path');
const fs      = require('fs');
const fetch   = require('node-fetch');
const FormData = require('form-data');
const db      = require('../config/db');

const AI_URL = process.env.AI_SERVICE_URL || 'http://localhost:5001';

async function callAIService(imageBuffer, mimeType = 'image/jpeg') {
  const form = new FormData();
  form.append('image', imageBuffer, { filename: 'upload.jpg', contentType: mimeType });

  const response = await fetch(`${AI_URL}/predict`, { method: 'POST', body: form });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`AI service error ${response.status}: ${text}`);
  }
  return response.json();
}

async function savePrediction(userId, imagePath, aiResult) {
  const n = aiResult.nutrition || {};
  const vitaminsJson = JSON.stringify({
    vitamin_c_mg:  n.vitamin_c_mg  ?? null,
    vitamin_a_iu:  n.vitamin_a_iu  ?? null,
    calcium_mg:    n.calcium_mg    ?? null,
    iron_mg:       n.iron_mg       ?? null,
    magnesium_mg:  n.magnesium_mg  ?? null,
    potassium_mg:  n.potassium_mg  ?? null,
    zinc_mg:       n.zinc_mg       ?? null,
    fiber_g:       n.fiber_g       ?? null,
    sugar_g:       n.sugar_g       ?? null,
  });

  const [result] = await db.query(
    `INSERT INTO predictions
       (user_id, image_path, fruit_name, confidence, ripeness,
        calories, carbohydrates, protein, fat, vitamins)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      userId,
      imagePath,
      aiResult.fruit_name,
      aiResult.confidence,
      aiResult.ripeness || null,
      n.energy_kcal   ?? null,
      n.carbs_g       ?? null,
      n.protein_g     ?? null,
      n.fat_g         ?? null,
      vitaminsJson,
    ]
  );
  return result.insertId;
}

// POST /api/predictions/upload  (multipart image)
exports.uploadPredict = async (req, res) => {
  try {
    if (!req.file)
      return res.status(400).json({ success: false, message: 'No image uploaded.' });

    const imageBuffer = req.file.buffer;
    const mimeType    = req.file.mimetype;
    const aiResult    = await callAIService(imageBuffer, mimeType);

    if (!aiResult.success)
      return res.status(422).json({ success: false, message: aiResult.message || 'No fruit detected.' });

    // Save image to disk
    const filename  = `${Date.now()}_${req.file.originalname.replace(/[^a-zA-Z0-9.]/g, '_')}`;
    const uploadDir = path.join(__dirname, '../../uploads');
    if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir, { recursive: true });
    const filePath = path.join(uploadDir, filename);
    fs.writeFileSync(filePath, imageBuffer);

    const predId = await savePrediction(req.user.id, `/uploads/${filename}`, aiResult);

    // Fetch saved record for WS broadcast
    const [rows] = await db.query(
      `SELECT p.*, u.name as user_name FROM predictions p
       JOIN users u ON p.user_id = u.id
       WHERE p.id = ?`, [predId]
    );
    const record = rows[0];

    // Emit via Socket.IO
    const io = req.app.get('io');
    if (io) {
      io.to(`user_${req.user.id}`).emit('new_prediction', record);
    }

    res.json({ success: true, prediction: record });
  } catch (err) {
    console.error('[Predict] uploadPredict error:', err);
    res.status(500).json({ success: false, message: 'Prediction failed: ' + err.message });
  }
};

// POST /api/predictions/webcam  (base64 JSON)
exports.webcamPredict = async (req, res) => {
  try {
    const { image_base64 } = req.body;
    if (!image_base64)
      return res.status(400).json({ success: false, message: 'No image data provided.' });

    // Strip data URI prefix
    let b64 = image_base64;
    if (b64.includes(',')) b64 = b64.split(',')[1];
    const imageBuffer = Buffer.from(b64, 'base64');

    const aiResult = await callAIService(imageBuffer, 'image/jpeg');

    if (!aiResult.success)
      return res.status(422).json({ success: false, message: aiResult.message || 'No fruit detected.' });

    // Save image to disk
    const filename  = `webcam_${Date.now()}.jpg`;
    const uploadDir = path.join(__dirname, '../../uploads');
    if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir, { recursive: true });
    fs.writeFileSync(path.join(uploadDir, filename), imageBuffer);

    const predId = await savePrediction(req.user.id, `/uploads/${filename}`, aiResult);

    const [rows] = await db.query(
      `SELECT p.*, u.name as user_name FROM predictions p
       JOIN users u ON p.user_id = u.id
       WHERE p.id = ?`, [predId]
    );
    const record = rows[0];

    const io = req.app.get('io');
    if (io) io.to(`user_${req.user.id}`).emit('new_prediction', record);

    res.json({ success: true, prediction: record });
  } catch (err) {
    console.error('[Predict] webcamPredict error:', err);
    res.status(500).json({ success: false, message: 'Prediction failed: ' + err.message });
  }
};
