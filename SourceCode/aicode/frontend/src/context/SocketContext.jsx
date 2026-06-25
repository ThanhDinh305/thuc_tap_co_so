import { createContext, useContext, useEffect, useRef, useState, useCallback } from 'react'
import { io } from 'socket.io-client'
import { useAuth } from './AuthContext'

const SocketContext = createContext(null)

export function SocketProvider({ children }) {
  const { token } = useAuth()
  const socketRef = useRef(null)
  const [connected, setConnected] = useState(false)
  const listeners = useRef({})  // eventName → Set of callbacks

  useEffect(() => {
    if (!token) {
      if (socketRef.current) {
        socketRef.current.disconnect()
        socketRef.current = null
        setConnected(false)
      }
      return
    }

    const socket = io('http://localhost:5000', {
      auth: { token },
      transports: ['websocket'],
      reconnection: true,
      reconnectionDelay: 2000,
    })

    socket.on('connect',    () => setConnected(true))
    socket.on('disconnect', () => setConnected(false))

    // Forward any registered events
    const ALL_EVENTS = ['new_prediction']
    ALL_EVENTS.forEach(event => {
      socket.on(event, data => {
        const cbs = listeners.current[event]
        if (cbs) cbs.forEach(cb => cb(data))
      })
    })

    socketRef.current = socket
    return () => { socket.disconnect() }
  }, [token])

  const subscribe = useCallback((event, callback) => {
    if (!listeners.current[event]) listeners.current[event] = new Set()
    listeners.current[event].add(callback)
    return () => listeners.current[event]?.delete(callback)
  }, [])

  return (
    <SocketContext.Provider value={{ connected, subscribe }}>
      {children}
    </SocketContext.Provider>
  )
}

export function useSocket() {
  return useContext(SocketContext)
}
