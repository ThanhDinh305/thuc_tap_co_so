const jwt = require('jsonwebtoken');

/**
 * Middleware: verify JWT from Authorization: Bearer <token>
 * Attaches req.user = { id, name, email, role }
 */
function authMiddleware(req, res, next) {
  const header = req.headers.authorization || '';
  if (!header.startsWith('Bearer ')) {
    return res.status(401).json({ success: false, message: 'No token provided.' });
  }
  const token = header.slice(7);
  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = decoded;
    next();
  } catch (err) {
    return res.status(401).json({ success: false, message: 'Invalid or expired token.' });
  }
}

/**
 * Middleware: require admin role
 * Must be used AFTER authMiddleware
 */
function adminMiddleware(req, res, next) {
  if (req.user?.role !== 'admin') {
    return res.status(403).json({ success: false, message: 'Admin access required.' });
  }
  next();
}

module.exports = { authMiddleware, adminMiddleware };
