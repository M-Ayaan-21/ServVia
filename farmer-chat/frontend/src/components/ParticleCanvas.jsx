import { useEffect, useRef } from 'react'

export default function ParticleCanvas() {
  const canvasRef = useRef(null)

  useEffect(() => {
    const c = canvasRef.current
    const ctx = c.getContext('2d')
    let w, h, particles = [], lastFrame = 0, animId
    const INTERVAL = 1000 / 30

    function resize() {
      w = c.width = window.innerWidth
      h = c.height = window.innerHeight
    }
    window.addEventListener('resize', resize)
    resize()

    class P {
      constructor() { this.reset() }
      reset() {
        this.x = Math.random() * w
        this.y = Math.random() * h
        this.r = Math.random() * 1.5 + 0.3
        this.vx = (Math.random() - 0.5) * 0.2
        this.vy = (Math.random() - 0.5) * 0.2
        this.a = Math.random() * 0.2 + 0.04
      }
      update() {
        this.x += this.vx
        this.y += this.vy
        if (this.x < 0 || this.x > w) this.vx *= -1
        if (this.y < 0 || this.y > h) this.vy *= -1
      }
      draw() {
        ctx.beginPath()
        ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(220,38,38,${this.a})`
        ctx.fill()
      }
    }

    for (let i = 0; i < 20; i++) particles.push(new P())

    function animate(ts) {
      animId = requestAnimationFrame(animate)
      if (ts - lastFrame < INTERVAL) return
      lastFrame = ts
      ctx.clearRect(0, 0, w, h)
      particles.forEach(p => { p.update(); p.draw() })
    }
    animId = requestAnimationFrame(animate)

    return () => {
      cancelAnimationFrame(animId)
      window.removeEventListener('resize', resize)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none"
      style={{ zIndex: 0 }}
    />
  )
}
