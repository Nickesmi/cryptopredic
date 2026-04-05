export const metadata = {
  title: 'CryptoPredic',
  description: 'AI-Powered Cryptocurrency Price Predictions and High Potential Anomalies Tracker.',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
