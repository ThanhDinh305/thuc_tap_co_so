const bcrypt  = require('bcryptjs');
const jwt     = require('jsonwebtoken');
const db      = require('../config/db');

function signToken(user) {
  return jwt.sign(
    { id: user.id, name: user.name, email: user.email, role: user.role },
    process.env.JWT_SECRET,
    { expiresIn: process.env.JWT_EXPIRES_IN || '7d' }
  );
}

// POST /api/auth/register
exports.register = async (req, res) => {
  try {
    const { name, email, password } = req.body;
    if (!name || !email || !password)
      return res.status(400).json({ success: false, message: 'All fields are required.' });
    if (password.length < 6)
      return res.status(400).json({ success: false, message: 'Password must be at least 6 characters.' });

    const [existing] = await db.query('SELECT id FROM users WHERE email = ?', [email]);
    if (existing.length > 0)
      return res.status(409).json({ success: false, message: 'Email already registered.' });

    const hashed = await bcrypt.hash(password, 10);
    const [result] = await db.query(
      'INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
      [name.trim(), email.toLowerCase().trim(), hashed]
    );

    const user = { id: result.insertId, name: name.trim(), email: email.toLowerCase().trim(), role: 'user' };
    const token = signToken(user);
    res.status(201).json({ success: true, token, user });
  } catch (err) {
    console.error('[Auth] register error:', err);
    res.status(500).json({ success: false, message: 'Server error.' });
  }
};

// POST /api/auth/login
exports.login = async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password)
      return res.status(400).json({ success: false, message: 'Email and password are required.' });

    const [rows] = await db.query('SELECT * FROM users WHERE email = ?', [email.toLowerCase().trim()]);
    if (rows.length === 0)
      return res.status(401).json({ success: false, message: 'Invalid credentials.' });

    const user = rows[0];
    const valid = await bcrypt.compare(password, user.password);
    if (!valid)
      return res.status(401).json({ success: false, message: 'Invalid credentials.' });

    const token = signToken(user);
    const { password: _, ...safeUser } = user;
    res.json({ success: true, token, user: safeUser });
  } catch (err) {
    console.error('[Auth] login error:', err);
    res.status(500).json({ success: false, message: 'Server error.' });
  }
};

// GET /api/auth/me
exports.getMe = async (req, res) => {
  try {
    const [rows] = await db.query(
      'SELECT id, name, email, role, avatar, created_at FROM users WHERE id = ?',
      [req.user.id]
    );
    if (rows.length === 0)
      return res.status(404).json({ success: false, message: 'User not found.' });
    res.json({ success: true, user: rows[0] });
  } catch (err) {
    res.status(500).json({ success: false, message: 'Server error.' });
  }
};

// PUT /api/auth/profile
exports.updateProfile = async (req, res) => {
  try {
    const { name, currentPassword, newPassword } = req.body;
    const [rows] = await db.query('SELECT * FROM users WHERE id = ?', [req.user.id]);
    if (rows.length === 0)
      return res.status(404).json({ success: false, message: 'User not found.' });

    const user = rows[0];
    const updates = {};
    if (name) updates.name = name.trim();

    if (newPassword) {
      if (!currentPassword)
        return res.status(400).json({ success: false, message: 'Current password required.' });
      const valid = await bcrypt.compare(currentPassword, user.password);
      if (!valid)
        return res.status(401).json({ success: false, message: 'Current password incorrect.' });
      if (newPassword.length < 6)
        return res.status(400).json({ success: false, message: 'New password must be at least 6 characters.' });
      updates.password = await bcrypt.hash(newPassword, 10);
    }

    if (Object.keys(updates).length === 0)
      return res.status(400).json({ success: false, message: 'Nothing to update.' });

    const setClauses = Object.keys(updates).map(k => `${k} = ?`).join(', ');
    await db.query(`UPDATE users SET ${setClauses} WHERE id = ?`, [...Object.values(updates), req.user.id]);

    const [updated] = await db.query(
      'SELECT id, name, email, role, avatar, created_at FROM users WHERE id = ?',
      [req.user.id]
    );
    res.json({ success: true, user: updated[0] });
  } catch (err) {
    console.error('[Auth] updateProfile error:', err);
    res.status(500).json({ success: false, message: 'Server error.' });
  }
};
