require('dotenv').config();
const express  = require('express');
const cors     = require('cors');
const path     = require('path');
const http     = require('http');
const { Server } = require('socket.io');

const app    = express();
const server = http.createServer(app);
const io     = new Server(server, {
  cors: { origin: ['http://localhost:5173', 'http://localhost:3000'], methods: ['GET', 'POST'] }
});

app.set('io', io);

// ── Middleware ────────────────────────────────────────────────────────────────
app.use(cors({ origin: ['http://localhost:5173', 'http://localhost:3000'], credentials: true }));
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Serve uploaded images
app.use('/uploads', express.static(path.join(__dirname, '../uploads')));

// ── Routes ────────────────────────────────────────────────────────────────────
app.use('/api/auth',        require('./routes/auth'));
app.use('/api/predictions', require('./routes/predictions'));
app.use('/api/history',     require('./routes/history'));
app.use('/api/dashboard',   require('./routes/dashboard'));
app.use('/api/admin',       require('./routes/admin'));

app.get('/api/health', (req, res) => res.json({ status: 'ok', timestamp: new Date().toISOString() }));

// 404 catch
app.use((req, res) => res.status(404).json({ success: false, message: `Route ${req.path} not found.` }));

// Error handler
app.use((err, req, res, next) => {
  console.error('[Server Error]', err.message);
  res.status(500).json({ success: false, message: err.message || 'Internal server error.' });
});

// ── Socket.IO ─────────────────────────────────────────────────────────────────
const jwt = require('jsonwebtoken');

io.use((socket, next) => {
  const token = socket.handshake.auth?.token;
  if (!token) return next(new Error('Authentication required'));
  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    socket.user = decoded;
    next();
  } catch {
    next(new Error('Invalid token'));
  }
});

io.on('connection', socket => {
  const userId = socket.user?.id;
  if (userId) {
    socket.join(`user_${userId}`);
    console.log(`[WS] User ${userId} connected (socket: ${socket.id})`);
  }

  socket.on('disconnect', () => {
    console.log(`[WS] Socket ${socket.id} disconnected`);
  });
});

// ── Start ─────────────────────────────────────────────────────────────────────
const PORT = parseInt(process.env.PORT || '5000');
server.listen(PORT, () => {
  console.log(`\n[Server] ✓ Running on http://localhost:${PORT}`);
  console.log(`[Server] ✓ Socket.IO ready`);
});
