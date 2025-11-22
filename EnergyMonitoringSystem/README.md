# PAC3220 Prepaid Energy Monitoring System

A comprehensive prepaid energy monitoring and billing platform for Siemens PAC3220 analyzers.

## 🚀 Features

- **Real-time Monitoring**: Continuous polling of up to 800 Siemens PAC3220 devices
- **Prepaid Billing**: Automatic cutoff when energy limits are reached
- **Dual Tariff Support**: Separate tracking for grid and generator power
- **Web Dashboard**: Professional admin and user interfaces
- **Mobile Ready**: Responsive design for mobile devices
- **Secure API**: JWT-based authentication with role-based access
- **Automated Alerts**: Email notifications for low balance and cutoffs

## 🏗️ System Architecture

### Components
- **Backend API**: FastAPI with SQL Server stored procedures
- **Database**: Microsoft SQL Server with comprehensive schema
- **Frontend**: React with Tailwind CSS and Recharts
- **Poller**: Python Modbus TCP client for device communication

### Database Schema
- `app.Users` - User accounts and prepaid allocations
- `app.Analyzers` - Siemens PAC3220 device configurations
- `app.Readings` - Real-time sensor data (20+ parameters)
- `app.Allocations` - Prepaid credit transactions
- `app.Alerts` - System and user notifications
- `ops.Events` - System events and audit logs

## 📋 Prerequisites

- Windows 10/11 or Windows Server
- SQL Server 2019+ (Express edition works)
- Python 3.11+
- Node.js 16+ (for frontend development)
- Siemens PAC3220 devices on network

## 🛠️ Installation

### 1. Database Setup

Run the automated installer:
```bash
install.bat
```

Or manually:
```sql
sqlcmd -S localhost -U sa -P YourPassword -i scripts\database_init.sql
```

### 2. Backend Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your database credentials

# Start the API server
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### 4. Start Modbus Poller

```bash
python backend/modbus_poller.py
```

## 🔧 Configuration

### Environment Variables (.env)

```env
# Database
DB_SERVER=localhost
DB_NAME=PAC3220DB
DB_USER=sa
DB_PASSWORD=YourStrongPassword
DB_DRIVER=ODBC Driver 17 for SQL Server

# Modbus
MODBUS_HOST=192.168.10.2
MODBUS_PORT=502
MODBUS_UNIT_ID=1
MODBUS_TIMEOUT=5
MODBUS_POLL_INTERVAL=5

# Security
JWT_SECRET=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30

# Email (optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
```

## 📊 API Endpoints

### Authentication
- `POST /api/auth/login` - User login
- `POST /api/auth/refresh` - Token refresh

### Admin Endpoints
- `GET /api/admin/users` - List all users
- `POST /api/admin/users/{user_id}/recharge` - Recharge user
- `GET /api/admin/dashboard` - Admin dashboard
- `GET /api/admin/events` - System events

### User Endpoints
- `GET /api/user/dashboard` - User dashboard data
- `GET /api/user/alerts` - User alerts

## 🔌 Modbus Parameters

The system reads 20+ parameters from PAC3220 devices:

| Category | Parameters |
|----------|------------|
| Power | KW L1, KW L2, KW L3, KW Total |
| Energy | KWh L1, KWh L2, KWh L3, KWh Total |
| Voltage | VL1, VL2, VL3 |
| Current | IL1, IL2, IL3, ITotal |
| Other | Frequency, PF L1-3, PF Avg |

## 🎯 Default Credentials

**Admin User:**
- Username: `admin`
- Password: `Admin123!`

⚠️ **Change the default password immediately after first login!**

## 🚀 Deployment

### Production Setup

1. **Database**: Use SQL Server on Windows Server
2. **Backend**: Deploy with IIS or as Windows Service
3. **Frontend**: Build and serve static files
4. **Poller**: Run as Windows Service with auto-restart
5. **Firewall**: Open ports 80/443, block Modbus ports from WAN

### Docker Deployment (Alternative)

```bash
# Build and run with Docker Compose
docker-compose up -d
```

## 📈 Monitoring & Maintenance

- **Logs**: Check `backend/logs/` for poller and API logs
- **Database**: Monitor space usage and backup regularly
- **Alerts**: Configure email alerts for system issues
- **Updates**: Apply firmware updates to PAC3220 devices

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check SQL Server service is running
   - Verify connection string in .env
   - Ensure firewall allows SQL Server port

2. **Modbus Connection Failed**
   - Verify PAC3220 IP address and network connectivity
   - Check Modbus unit ID (default: 1)
   - Ensure no firewall blocking port 502

3. **Poller Not Reading Data**
   - Check device register mapping
   - Verify PAC3220 is powered and configured
   - Review poller logs for specific errors

## 📚 API Documentation

Access interactive API documentation at:
- `http://localhost:8000/docs` (Swagger UI)
- `http://localhost:8000/redoc` (ReDoc)

## 🤝 Support

For technical support or questions:
- Check the logs in `backend/logs/`
- Review database events in `ops.Events` table
- Ensure all prerequisites are properly installed

## 📄 License

This project is proprietary software for PAC3220 energy monitoring systems.

---

**Version**: 1.0.0
**Database Schema**: v1.0
**Last Updated**: 2025-11-14